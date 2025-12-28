import asyncio
import random
import threading
import fitz

import pdfplumber
import time
import re
import numpy as np
import math
import multiprocessing
from PIL import Image
from pypdf import PdfReader
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from .utils import OCR
from .recognizer import Recognizer
from .layout_recognizer import LayoutRecognizer4YOLOv10
from .table_structure_recognizer import TableStructureRecognizer
from apps.logs.logs import get_logger

logger = get_logger(__name__)


class PdfParser:
    """pdf解析器"""

    def __init__(self):
        self.zoomin = 3
        self.lefted_chars = []
        self.is_english = False
        self.imit_page_num = 32
        self.lock = None

    async def __call__(self, file_path, *args, **kwargs):
        logger.info("PdfParser init")
        logger.info("this file has images")
        self.filepath = file_path
        st = time.time()
        file_content = self.extract_pdf_structure()
        if file_content:
            logger.info(f"complete get_file_text, time: {time.time() - st}")
            return file_content
        self.get_file_images()
        self.page_chars = await self.get_file_chars()
        self.total_page = len(self.page_chars)
        self.get_file_outline()
        self.identify_language()
        logger.info(f"complete get_file_chars -> get_file_outline -> identify_language cost: {time.time() - st}")

        await self.get_boxes()  # det and rec
        logger.info(f"complete det and rec cost: {time.time() - st}, page_num:{len(self.boxes)}, "
                    f"boxes_num:{(len(self.boxes[0]) + len(self.boxes[1])) * len(self.boxes)}")
        self.layouts_rec()
        logger.info(f"complete layouts rec cost: {time.time() - st}, page_num:{len(self.boxes)}, "
                    f"boxes_num:{(len(self.boxes[0]) + len(self.boxes[1])) * len(self.boxes)}")
        self.table_transformer_job()
        self.text_merge()
        tables = self.extract_table_figure()
        self.naive_vertical_merge()
        self.filter_forpages()
        sections = self.merge_with_same_bullet()
        logger.info(f"complete table, images and text extract: {time.time() - st}, "
                    f"text chunk num:{len(sections)}, text keys:{list(sections[0].keys())}, "
                    f"tables|images num:{len(tables)}")
        sorted_blocked = self.get_text(sections, tables)
        return sorted_blocked

    def extract_pdf_structure(self):
        pages = fitz.open(self.filepath)
        has_images = any(page.get_images() for page in pages)
        if not has_images:
            loader = PyMuPDF4LLMLoader(self.filepath)
            docs = loader.load()
            content = '\n\n'.join(doc.page_content for doc in docs)

            return {"type": "paragraph", "text": content}
        return None

    def get_file_images(self, page_from=0, page_to=299):
        try:
            with pdfplumber.open(self.filepath) if isinstance(self.filepath, str) else pdfplumber.open(
                    BytesIO(self.filepath)) as pdf:
                # 默认 PDF 内部单位 = 72 DPI 216 DPI + antialias抗锯齿 图像更大，但识别更准（适合表格）
                # When initially converting the PDF into images，*self.zoomin
                self.page_images = [p.to_image(resolution=72 * self.zoomin, antialias=True).annotated for i, p in
                                    enumerate(pdf.pages[page_from:page_to])]
        except Exception as e:
            logger.error(f"PdfParser get_file_text failed :{e}")

    async def get_file_chars(self):
        if len(self.page_images) > self.imit_page_num:
            try:
                with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    tasks = [
                        asyncio.get_event_loop().run_in_executor(executor, self._parse_page, _)
                        for _ in range(math.ceil(len(self.page_images) / self.imit_page_num))
                    ]
                    page_chars = await asyncio.gather(*tasks)
                    page_chars_flatten = [item for sublist in page_chars for item in sublist]
                    return page_chars_flatten
            except Exception as e:
                logger.warning(f"Failed to extract characters for pages {0}-{len(self.page_images)}: {str(e)}")
                page_chars_flatten = [[] for _ in
                                      range(len(self.page_images))]  # If failed to extract, using empty list instead.
                return page_chars_flatten
        else:
            page_chars = self._parse_page(0)
            return page_chars

    def _parse_page(self, page_num):
        with pdfplumber.open(self.filepath) if isinstance(self.filepath, str) else pdfplumber.open(
                BytesIO(self.filepath)) as pdf:
            start = page_num * self.imit_page_num
            end = min((page_num + 1) * self.imit_page_num, len(pdf.pages))
            logger.info(f"parallel process parse page {start}:{end}")
            pages = pdf.pages[start: end]
            return [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in pages]

    def get_file_outline(self):
        self.outlines = []
        try:
            with PdfReader(self.filepath if isinstance(self.filepath, str) else BytesIO(self.filepath)) as pdf:
                outlines = pdf.outline

                def dfs(arr, depth):
                    for a in arr:
                        if isinstance(a, dict):
                            self.outlines.append((a["/Title"], depth))
                            continue
                        dfs(a, depth + 1)

                dfs(outlines, 0)
        except Exception as e:
            logger.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logger.info("Miss outlines")

    def identify_language(self):

        def _is_english(text):
            random_chars = random.choices([c["text"] for c in text], k=min(100, len(text)))
            result = re.search(r"[a-zA-Z0-9,/¸;:'\[\]()!@#$%^&*\"?<>._-]{30,}", ''.join(random_chars))
            return 1 if result else 0

        is_english = [_is_english(page) for page in self.page_chars]
        self.is_english = True if sum(is_english) / len(is_english) > 0.5 else False

    async def get_boxes(self):
        """
        return:
        [page_num, boxes_num, box_property_num]
        Property:
            {
                'bottom': np.float32(94.666664), 文本框底部坐标
                'page_number': 0, 文本框页数
                'text':'电力系统课程设计（学生用）', 文本框文本
                'top': np.float32(79.666664), 文本框顶部坐标
                'x0': np.float32(206.33333), 文本框左坐标
                'x1': np.float32(381.0) 文本框右坐标
            }
        """
        self.mean_height = [0.0] * self.total_page
        self.mean_width = [0.0] * self.total_page
        self.page_cum_height = [0.0] * self.total_page
        self.boxes = [[] for _ in range(self.total_page)]
        self.lock = threading.Lock()
        n = 4  # multiprocessing.cpu_count()
        ocr_instances = [OCR() for _ in range(n)]
        page_images_list = [self.page_images[i:i + n] for i in range(0, self.total_page, n)]
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=n) as executor:
            for start_idx, page_images in enumerate(page_images_list):
                tasks = []
                for rel_idx, img in enumerate(page_images):
                    global_page_num = start_idx * n + rel_idx  # 修正为全局页码
                    # 分配独立的OCR实例
                    if global_page_num >= self.total_page:
                        continue
                    ocr = ocr_instances[rel_idx % len(ocr_instances)]
                    # 提交任务时携带全局页码
                    task = loop.run_in_executor(
                        executor, self._det_and_rec_boxes, global_page_num, img, ocr
                    )
                    tasks.append(task)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=False)

        if not self.is_english and not any(self.page_chars) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                                        "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        self.page_cum_height = np.cumsum([0] + self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        for ocr in ocr_instances:
            if hasattr(ocr, "close"):
                ocr.close()
        self.lock = None
        if len(self.boxes) == 0 and self.zoomin < 9:
            self.zoomin = self.boxes * 3
            await self.get_boxes()
        self.zoomin = 3

    def _det_and_rec_boxes(self, page_num, img, ocr):
        with self.lock:
            chars = self.page_chars[page_num] if not self.is_english else []

        def safe_div(a, b, default=0):
            return float(a / b) if b != 0 else float(default)

        def safe_median(vals, default=0):
            valid_vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
            return np.median(valid_vals) if valid_vals else default

        mean_height = safe_median([c["height"] for c in chars], default=0)
        mean_width = safe_median([c["width"] for c in chars], default=8)
        cum_height = safe_div(img.size[1], self.zoomin)
        with self.lock:
            self.mean_height[page_num] = mean_height
            self.mean_width[page_num] = mean_width
            self.page_cum_height[page_num] = cum_height

        # Add space between English characters
        for j in range(len(chars) - 1):
            merge_chars = chars[j]["text"] + chars[j + 1]["text"]
            if (
                    merge_chars.strip()
                    and re.match(r"[0-9a-zA-Z,.:;!%]+", merge_chars)
                    and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
            ):
                chars[j]["text"] += " "

        np_img = np.array(img)
        # Use the detect model
        bxs = ocr.detect(np_img)
        if not bxs:
            logger.warning("No boxes detected")
            with self.lock:
                self.boxes[page_num] = []
            return
        bxs = [(line[0], line[1][0]) for line in bxs]

        bxs = Recognizer.sort_Y_firstly(
            [
                {"x0": b[0][0] / self.zoomin, "x1": b[1][0] / self.zoomin, "top": b[0][1] / self.zoomin, "text": "",
                 "txt": t, "bottom": b[-1][1] / self.zoomin, "chars": [], "page_number": page_num + 1}
                for b, t in bxs
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[page_num] / self.zoomin,
        )  # Get the real coordinates of the text box，/ self.zoomin

        # merge chars in the same rect
        for c in chars:
            with self.lock:  # 加锁保护lefted_chars
                ii = Recognizer.find_overlapped(c, bxs)
                if ii is None:
                    self.lefted_chars.append(c)
                    continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            # 修复零除：max(ch, bh) 先判断是否为0
            height_ratio = abs(ch - bh) / max(ch, bh) if max(ch, bh) != 0 else 1.0
            if height_ratio >= 0.7 and c["text"].strip() != "":  # 修复空字符串判断
                with self.lock:
                    self.lefted_chars.append(c)
                continue
            bxs[ii]["chars"].append(c)
        for b in bxs:
            if "chars" in b and not b["chars"]:
                del b["chars"]
                continue
            # 修复空chars的均值计算
            if "chars" in b and b["chars"]:
                m_ht = safe_median([c["height"] for c in b["chars"]], default=0)
                for c in Recognizer.sort_Y_firstly(b["chars"], m_ht):
                    if c["text"].strip() == " " and b["text"].strip():
                        # 修复正则匹配：锚定末尾，避免误加空格
                        if re.search(r"[0-9a-zA-Zа-яА-Я,.?;:!%]$", b["text"]):
                            b["text"] += " "
                    else:
                        b["text"] += c["text"].strip()  # 去除首尾空格
                del b["chars"]

        boxes_to_reg = []
        for b in bxs:
            if not b["text"].strip():
                # 安全计算坐标（防零除）
                left = safe_div(b["x0"], 1 / self.zoomin, default=0)  # 等价于 b["x0"] * self.zoomin
                right = safe_div(b["x1"], 1 / self.zoomin, default=0)
                top = safe_div(b["top"], 1 / self.zoomin, default=0)
                bott = safe_div(b["bottom"], 1 / self.zoomin, default=0)
                # 修复图片裁剪：校验坐标合法性
                if left < right and top < bott:
                    try:
                        b["box_image"] = ocr.get_rotate_crop_image(
                            np_img,
                            np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32)
                        )
                        boxes_to_reg.append(b)
                    except Exception as e:
                        logger.error("Page %s 裁剪文本框图片失败：%s", page_num, e)
            # 安全删除key：先判断是否存在
            if "txt" in b:
                del b["txt"]

        # 修复识别结果长度不匹配：先校验列表非空
        texts = []
        if boxes_to_reg:
            try:
                texts = ocr.recognize_batch([b["box_image"] for b in boxes_to_reg])
            except Exception as e:
                logger.error("Page %s OCR批量识别失败：%s", page_num, e)
                texts = [""] * len(boxes_to_reg)  # 兜底空字符串

        # 修复索引错位：按最小长度遍历
        for i in range(min(len(boxes_to_reg), len(texts))):
            boxes_to_reg[i]["text"] = texts[i].strip()
            if "box_image" in boxes_to_reg[i]:
                del boxes_to_reg[i]["box_image"]  # 释放图片内存

        # 过滤空文本框
        bxs = [b for b in bxs if b["text"].strip()]
        with self.lock:
            if self.mean_height[page_num] == 0 and bxs:
                self.mean_height[page_num] = safe_median([b["bottom"] - b["top"] for b in bxs], default=0)
            self.boxes[page_num] = bxs

    def layouts_rec(self, drop=True):
        layouter = LayoutRecognizer4YOLOv10()
        assert len(self.page_images) == len(self.boxes)
        self.boxes, self.page_layout = layouter(self.page_images, self.boxes, self.zoomin, drop=drop)
        # cumlative Y
        for i in range(len(self.boxes)):
            self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def table_transformer_job(self):
        imgs, pos = [], []
        tbcnt = [0]
        MARGIN = 10
        self.tb_cpns = []
        assert len(self.page_layout) == len(self.page_images)
        for p, tbls in enumerate(self.page_layout):  # for page
            tbls = [f for f in tbls if f["type"] == "table"]
            tbcnt.append(len(tbls))
            if not tbls:
                continue
            for tb in tbls:  # for table
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                left *= self.zoomin
                top *= self.zoomin
                right *= self.zoomin
                bott *= self.zoomin
                pos.append((left, top))
                imgs.append(self.page_images[p].crop((left, top, right, bott)))
        assert len(self.page_images) == len(tbcnt) - 1
        if not imgs:
            return
        tbl_det = TableStructureRecognizer()
        recos = tbl_det(imgs)
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # for page
            pg = []
            for j, tb_items in enumerate(recos[tbcnt[i]: tbcnt[i + 1]]):  # for table
                poss = pos[tbcnt[i]: tbcnt[i + 1]]
                for it in tb_items:  # for table components
                    it["x0"] = it["x0"] + poss[j][0]
                    it["x1"] = it["x1"] + poss[j][0]
                    it["top"] = it["top"] + poss[j][1]
                    it["bottom"] = it["bottom"] + poss[j][1]
                    for n in ["x0", "x1", "top", "bottom"]:
                        it[n] /= self.zoomin
                    it["top"] += self.page_cum_height[i]
                    it["bottom"] += self.page_cum_height[i]
                    it["pn"] = i
                    it["layoutno"] = j
                    pg.append(it)
            self.tb_cpns.extend(pg)

    def text_merge(self):
        # merge adjusted boxes
        bxs = self._assign_column(self.boxes)

        # horizontally merge adjacent box with the same layout
        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]

            if b["page_number"] != b_["page_number"] or b.get("col_id") != b_.get("col_id"):
                i += 1
                continue

            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure",
                                                                                                 "equation"]:
                i += 1
                continue

            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
                # merge
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
        self.boxes = bxs

    @staticmethod
    def _assign_column(boxes):
        if not boxes:
            return boxes
        if all("col_id" in b for b in boxes):
            return boxes

        by_page = defaultdict(list)
        for b in boxes:
            by_page[b["page_number"]].append(b)

        page_cols = {}

        for pg, bxs in by_page.items():
            if not bxs:
                page_cols[pg] = 1
                continue

            x0s_raw = np.array([b["x0"] for b in bxs], dtype=float)

            min_x0 = np.min(x0s_raw)
            max_x1 = np.max([b["x1"] for b in bxs])
            width = max_x1 - min_x0

            INDENT_TOL = width * 0.12
            x0s = []
            for x in x0s_raw:
                if abs(x - min_x0) < INDENT_TOL:
                    x0s.append([min_x0])
                else:
                    x0s.append([x])
            x0s = np.array(x0s, dtype=float)

            max_try = min(4, len(bxs))
            if max_try < 2:
                max_try = 1
            best_k = 1
            best_score = -1

            for k in range(1, max_try + 1):
                km = KMeans(n_clusters=k, n_init="auto")
                labels = km.fit_predict(x0s)

                centers = np.sort(km.cluster_centers_.flatten())
                if len(centers) > 1:
                    try:
                        score = silhouette_score(x0s, labels)
                    except ValueError:
                        continue
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_k = k
                if k == len(np.unique(x0s)):
                    break

            page_cols[pg] = best_k

        for pg, bxs in by_page.items():
            if not bxs:
                continue
            k = page_cols[pg]
            if len(bxs) < k:
                k = 1
            x0s = np.array([[b["x0"]] for b in bxs], dtype=float)
            km = KMeans(n_clusters=k, n_init="auto")
            labels = km.fit_predict(x0s)

            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)

            remap = {orig: new for new, orig in enumerate(order)}

            for b, lb in zip(bxs, labels):
                b["col_id"] = remap[lb]

            grouped = defaultdict(list)
            for b in bxs:
                grouped[b["col_id"]].append(b)

        return boxes

    @staticmethod
    def _y_dis(a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    @staticmethod
    def _x_dis(a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]), abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def extract_table_figure(self, need_image=True, need_position=True, separate_tables_figures=False):
        tables = {}
        figures = {}
        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption",
                                                                                                      "title",
                                                                                                      "figure caption",
                                                                                                      "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # find captions and pop out
        i = 0
        while i < len(self.boxes):
            c = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            # find the nearest layouts
            def nearest(tbls):
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logger.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            elif fk:
                figures[fk].insert(0, c)
                logger.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            ZM = self.zoomin
            pn = set([b["page_number"] - 1 for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {"x0": np.min([b["x0"] for b in bxs]), "top": np.min([b["top"] for b in bxs]) - ht,
                     "x1": np.max([b["x1"] for b in bxs]), "bottom": np.max([b["bottom"] for b in bxs]) - ht}
                louts = [layout for layout in self.page_layout[pn] if layout["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                # else:
                #     logger.warning(f"Missing layout match: {pn + 1},%s" % (bxs[0].get("layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left:
                    right = left + 1
                poss.append((pn, left, right, top, bott))
                return self.page_images[pn].crop((left * ZM, top * ZM, right * ZM, bott * ZM))
            pn = {}
            for b in bxs:
                p = b["page_number"] - 1
                if p not in pn:
                    pn[p] = []
                pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            pic = Image.new("RGB", (int(np.max([i.size[0] for i in imgs])), int(np.sum([m.size[1] for m in imgs]))),
                            (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # crop figure out and add caption
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []

            if separate_tables_figures:
                figure_results.append((cropout(bxs, "figure", poss), [txt]))
                figure_positions.append(poss)
            else:
                res.append((cropout(bxs, "figure", poss), [txt]))
                positions.append(poss)

        tbl_det = TableStructureRecognizer()

        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs]))

            poss = []

            res.append((cropout(bxs, "table", poss),
                        tbl_det.construct_table(bxs, html=True, is_english=self.is_english)))
            positions.append(poss)

        if separate_tables_figures:
            assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
            if need_position:
                return list(zip(res, positions)), list(zip(figure_results, figure_positions))
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                return list(zip(res, positions))
            else:
                return res

    def naive_vertical_merge(self):
        bxs = self._assign_column(self.boxes)  # 获得列id

        grouped = defaultdict(list)
        for b in bxs:
            grouped[(b["page_number"], b.get("col_id", 0))].append(b)

        merged_boxes = []
        for (pg, col), bxs in grouped.items():
            bxs = sorted(bxs, key=lambda x: (x["top"], x["x0"]))
            if not bxs:
                continue

            mh = self.mean_height[pg - 1] if self.mean_height else np.median(
                [b["bottom"] - b["top"] for b in bxs]) or 10

            i = 0
            while i + 1 < len(bxs):
                b = bxs[i]
                b_ = bxs[i + 1]

                if b["page_number"] < b_["page_number"] and re.match(r"[0-9 •一—-]+$", b["text"]):
                    bxs.pop(i)
                    continue

                if not b["text"].strip():
                    bxs.pop(i)
                    continue

                if not b["text"].strip() or b.get("layoutno") != b_.get("layoutno"):
                    i += 1
                    continue

                if b_["top"] - b["bottom"] > mh * 1.5:
                    i += 1
                    continue

                overlap = max(0, min(b["x1"], b_["x1"]) - max(b["x0"], b_["x0"]))
                if overlap / max(1, min(b["x1"] - b["x0"], b_["x1"] - b_["x0"])) < 0.3:
                    i += 1
                    continue

                concatting_feats = [
                    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                    len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，‘“、；：",
                    b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",
                ]
                # features for not concating
                feats = [
                    b.get("layoutno", 0) != b_.get("layoutno", 0),
                    b["text"].strip()[-1] in "。？！?",
                    self.is_english and b["text"].strip()[-1] in ".!?",
                    b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[
                        b["page_number"] - 1] * 1.5,
                    b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[
                        b["page_number"] - 1] * 4,
                ]
                # split features
                detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]
                if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                    logger.debug(
                        "{} {} {} {}".format(
                            b["text"],
                            b_["text"],
                            any(feats),
                            any(concatting_feats),
                        )
                    )
                    i += 1
                    continue

                b["text"] = (b["text"].rstrip() + " " + b_["text"].lstrip()).strip()
                b["bottom"] = b_["bottom"]
                b["x0"] = min(b["x0"], b_["x0"])
                b["x1"] = max(b["x1"], b_["x1"])
                bxs.pop(i + 1)

            merged_boxes.extend(bxs)

        self.boxes = sorted(merged_boxes, key=lambda x: (x["page_number"], x.get("col_id", 0), x["top"]))

    def filter_forpages(self):
        if not self.boxes:
            return
        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                            re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            findit = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(
                self.boxes[i]["text"].strip().split()[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(
                    self.boxes[i]["text"].strip().split()[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def merge_with_same_bullet(self):
        def _is_chinese(s):
            if "\u4e00" <= s <= "\u9fa5":
                return True
            else:
                return False

        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if (
                    b["text"].strip()[0] != b_["text"].strip()[0]
                    or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                    or _is_chinese(b["text"].strip()[0])
                    or b["top"] > b_["bottom"]
            ):
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)
        return self.boxes

    def get_text(self, sections, tables):
        blocks = []
        for section in sections:
            block = {}
            block['x0'] = section['x0']
            block['top'] = section['top']
            block['text'] = section['text']
            block['type'] = 'title' if section['layout_type'] == 'title' else 'paragraph'
            blocks.append(block)
        for table in tables:
            block = {}
            block['x0'] = table[1][0][1]
            block['top'] = self.page_cum_height[table[1][0][0]] + float(table[1][0][3])
            text = table[0][1]
            if isinstance(text, list):
                block['text'] = '\n'.join(text)
                block['type'] = 'images'
            else:
                block['text'] = text
                block['type'] = 'table'
            blocks.append(block)
        sorted_blocked = Recognizer.sort_Y_firstly(
            blocks,
            sum(self.mean_height[page_num] / self.zoomin for page_num in range(len(self.mean_height))) / len(
                self.mean_height),
        )
        return sorted_blocked

    @staticmethod
    def _has_color(o):
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and \
                    o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]()-]+", o.get("text", "")):
                    return False
        return True
