# 读取xlsx、xls
- 读取表格sheet名
- 根据sheet名获取sheet页内容
- 转化为md，进行拼接
# 读取txt
- 直接读取，txt文档没有格式
# 读取pdf（借鉴deepdoc）
- 获取pdf文本框、表格、图片
## 获取pdf文本 将pdf转化为图片
## 获取目录
## 图片识别
### 一.获取单页的平均高度 平均宽度 还有按zoomin缩放后的高度 img.size[1] / zoomin 
- 平均高度
- 平均宽度
- 按zoomin缩放后的高度
### 二.循环获取每个字符文本框的字符，对两个相邻的文本块添加空格
这个的场景应该是模拟判断条件
- 当前文本不为空
- 下一文本不为空
- 当前和下一文本中含英文字符或文本常用标点 re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
- 当前文本和下一文本横坐标位置的间隔大于min(width1, width2)的1/2\
满足的场景应该是两个单字符：A B；或者多个字符中间相隔很远：hello      world
### 三、检测
#### 1.对图片进行transform
- 把图片转化为最长边是960
- 归一化
- 把维度提到最前面
返回的是归一化后的图片，[原始高，原始宽，高缩放比例，宽缩放比例]\
图片tensor:[维度，高，宽]
#### 2.检测
- 变为四维张量[1，维度，高，宽]输入到det.onnx模型输出为[1，1，高，宽]
#### 3.后处理
- 张量维度变为[1，高，宽] 设阈值大于0.3判断张量中是否存在大于阈值的值 segmentation = pred > self.thresh
- 遍历维度（即[1，高，宽]中的1）获得掩码 mask = segmentation[batch_index]
- 发现边界
```python
outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
# bitmap 通常是二值化图像（像素值为 0 或 1，0 代表背景，1 代表前景），但 OpenCV 的轮廓检测要求输入图像是 **8 位无符号整数类型（np.uint8）** 且像素值为 0（黑）或 255（白）。
# 这可能就是为啥会对图片进行归一化
# 第二个参数：轮廓检索模式（cv2.RETR_LIST）：
# RETR_LIST 表示只检索所有轮廓，不建立轮廓间的层级关系（父子轮廓），是最简单的检索模式，适合仅需提取所有轮廓的场景。
# 第三个参数：轮廓逼近方法（cv2.CHAIN_APPROX_SIMPLE）：
# 用于压缩轮廓点的冗余信息。例如，对于矩形轮廓，它会只保留矩形的 4 个顶点，而非存储轮廓上所有点，从而节省内存。
# 输出 contours 是检测到的轮廓列表（每个轮廓是点的坐标数组），hierarchy 是轮廓层级信息。
```
- 对contours进行处理
```python
bounding_box = cv2.minAreaRect(contour)
# (center, (width, height), angle)  # 计算能包围轮廓的最小矩形
# center：矩形的中心点坐标 (x, y)；
# (width, height)：矩形的宽度和高度（对应最小面积的尺寸）；
# angle：矩形旋转的角度（与水平轴的夹角，范围 [-90, 0)）
points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
# cv2.boxPoints(bounding_box)：将 cv2.minAreaRect() 返回的矩形信息转换为四个顶点的坐标数组（格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]）。
# list(...)：将 OpenCV 输出的数组转为 Python 列表，方便后续排序。
# sorted(..., key=lambda x: x[0])：按顶点的横坐标（x[0]）从小到大排序四个顶点。
```
返回从左下角顺时针旋转的举行坐标，和矩形的min(width, height) 如果min(width, height)<3 抛弃
- 框出目标区域，只算框内多边形部分的像素平均值 self.box_thresh > score 抛弃
- 对目标区域进行扩张 变为1.5倍 扩展为12边形 再获得最小的四边形
- 将最后坐标映射到原始图片坐标 获得检测边框
#### 4.再次对文本块进行排序 先上下后左右的顺序
#### 5.将文本与文本框进行对应
- 二分法找对应的文本框 若找不到加入left块中
- 获得文本框 计算字符框高度ch与文本行框高度bh的差异比例：abs(ch - bh)/max(ch, bh)；
- 若高度差异比例≥0.7（即高度差异过大，字符与行不匹配）且字符不是空格，则将该字符框加入lefted_chars
- 若匹配成功且高度符合要求，将字符框c添加到对应文本行框bxs[ii]的chars列表中（实现字符与文本行的关联
#### 6.文本拼接
- 遍历文本行框并过滤空字符框
- 计算字符平均高度并排序字符框 统计当前文本行内所有字符的高度平均值，作为排序的阈值 对字符框按 “先垂直（Y 轴）、后水平（X 轴）” 排序
- 按规则拼接字符为文本 文本最后一个字符是字母/数字/标点 在字符后面加空格 否则不加空格\
<span style="color:#00ff00">**到这一步获得获得按行分割的文本**</span>
#### 进一步处理图像识别框
- 对于没有文本的文本框或者叫识别框进行提取并记录
```python
img_crop_width = int(
    max(
        np.linalg.norm(points[0] - points[1]),  # 上边长度
        np.linalg.norm(points[2] - points[3])))  # 下边长度
img_crop_height = int(
    max(
        np.linalg.norm(points[0] - points[3]),  # 左边长度
        np.linalg.norm(points[1] - points[2])))  # 右边长度
# 通过计算四边形对边的欧式距离，取最大值作为裁剪后的图像宽度和高度，确保完整覆盖目标区域
pts_std = np.float32([[0, 0], [img_crop_width, 0],
                      [img_crop_width, img_crop_height],
                      [0, img_crop_height]])
M = cv2.getPerspectiveTransform(points, pts_std)
dst_img = cv2.warpPerspective(
    img,
    M, (img_crop_width, img_crop_height),
    borderMode=cv2.BORDER_REPLICATE,
    flags=cv2.INTER_CUBIC)
# pts_std定义了裁剪后标准矩形的四个顶点（正矩形）。
# cv2.getPerspectiveTransform计算原始四边形points到标准矩形pts_std的透视变换矩阵M。
# cv2.warpPerspective执行透视变换，将倾斜的文本区域矫正为正矩形：
# borderMode=cv2.BORDER_REPLICATE：边界填充方式为复制边缘像素。
# flags=cv2.INTER_CUBIC：插值方式为双三次插值（保证图像质量）
if dst_img_height * 1.0 / dst_img_width >= 1.5:
    # 原始方向识别
    rec_result = self.text_recognizer[0]([dst_img])
    text, score = rec_result[0][0]
    best_score = score
    best_img = dst_img

    # 顺时针旋转90°（k=3等价于顺时针转90°）
    rotated_cw = np.rot90(dst_img, k=3)
    rec_result = self.text_recognizer[0]([rotated_cw])
    rotated_cw_text, rotated_cw_score = rec_result[0][0]
    if rotated_cw_score > best_score:
        best_score = rotated_cw_score
        best_img = rotated_cw

    # 逆时针旋转90°（k=1）
    rotated_ccw = np.rot90(dst_img, k=1)
    rec_result = self.text_recognizer[0]([rotated_ccw])
    rotated_ccw_text, rotated_ccw_score = rec_result[0][0]
    if rotated_ccw_score > best_score:
        best_img = rotated_ccw

    dst_img = best_img
# 当裁剪后图像的高宽比≥1.5（竖长图像）时，文本可能因方向问题导致识别准确率低，因此尝试不同旋转方向：
# 原始方向识别，记录最优分数best_score和对应图像best_img。
# 顺时针旋转 90° 后识别，若分数更高则更新最优结果。
# 逆时针旋转 90° 后识别，若分数更高则更新最优结果。
# 最终选择识别分数最高的旋转方向对应的图像作为输出。
```
- 重新识别文本记录到文本框
## 获取文本/实体框 文本与实体框相关联后获取文本框 -> 进行**layout识别**

### 预处理

- 规定输出的宽高维度为[1024,1024]，根据图片想对于输入宽高的最小比值，等比例缩放图片
- 转换色彩空间并调整数据类型，适配模型输入要求；
- 使用线性插值（INTER_LINEAR）缩放图像，平衡效率与质量。
- 填充固定颜色到输入要求tensor维度
- 归一化像素，转换维度，添加维度 -> NCHW
- 存储处理后的图像及缩放/填充参数（用于后续坐标映射）

### 利用模型进行版面识别

### 后处理

- 获取识别结果的置信度scores = boxes[:, 4] 过滤置信度小于阈值0.08的box
- 获取识别框类别class_ids = boxes[:, -1].astype(int)
- 还原图片 获取原始坐标，减去填充，乘以缩放因子
- 获取所有唯一类别ID 提取当前类别的所有检测框索引、坐标、置信度 对当前类别执行NMS（IoU阈值0.45），去除重叠框
- 获取结果类别名称（小写） 检测框坐标（转为float列表） 置信度

### 去除重叠的layout框

- 转化格式 将layout按照从上到下，左到右进行排序

```python
lts = [
    {
        "type": b["type"],
        "score": float(b["score"]),
        "x0": b["bbox"][0] / scale_factor,
        "x1": b["bbox"][2] / scale_factor,
        "top": b["bbox"][1] / scale_factor,
        "bottom": b["bbox"][-1] / scale_factor,
        "page_number": pn,
    }
    for b in lts
    if float(b["score"]) >= 0.4 or b["type"] not in self.garbage_layouts
]
```

- 遍历layout框，找附近两个layout框是否重叠，重叠但面积比例小于0.7不用管
- 如果重叠率大于0.7，保留置信度大的layout框。如果两个置信度相等，计算layout框和文本框之间的重叠面积，取重叠面积大的layout框。

### 将文本框和layout对应

- 遍历layout类型
  ["footer", "header", "reference", "figure caption", "table caption", "title", "table", "text", "figure", "equation"]
- 遍历文本框
- 判断当前文本框是否和当前layout框重叠度大于0.4 大于则继续
- 若当前layout框类型为页眉/页脚 且只占页面10%的面积 -> 判断为页眉页脚， 删除掉文本框
- 记录当前布局编号 布局类型
- 如果文本框类型为"figure"/"equation"且没有关联的文本框 -> OCR 识别阶段可能未生成对应的文本框，进行补充\
  记录layout框位置，添加其他键值对，并加入到文本框列表中

```python
del lt["type"]
lt["text"] = ""
lt["layout_type"] = "figure"
lt["layoutno"] = f"figure-{i}"
```

- 最后再做一遍ocr的净化\
  统计一下页眉页脚的文本，然后遍历文本框，获取文本内容，如果文本框中的内容都是页眉页脚的文本，则删除该文本框
- 记录每个文本框的全局位置，上边界/下边界

```python
for i in range(len(self.boxes)):
    self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
    self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
```

<span style="color:#aaffff">**截止到目前 文本框有8个属性 bottom layout_type layoutno page_number text top x0 x1**</span>
```json
{
  'bottom': np.float64(94.66666412353516),
  'layout_type': 'title',
  'layoutno': 'title-0',
  'page_number': 1,
  'text': '电力系统课程设计（学生用）',
  'top': np.float64(79.66666412353516),
  'x0': np.float32(206.33333),
  'x1': np.float32(381.0)
}
```
## 进行表格处理
### 获取表格所在位置的图片
- 遍历layout框找到标签为表格的layout框
- 向外扩展 然后还原原始图片位置进行裁剪
### 对图片进行预处理 报错缩放到输入尺寸，归一化 转化维度
### 对截取后的表格图片进行表格识别（获得表格的版面标签）
### 后处理
- 对每个类别独立执行 IoU 过滤，避免跨类别干扰
- 保留置信度高且重叠度低的检测框，消除同一目标的重复检测
- 通过设置 IoU 阈值（此处为 0.2）控制过滤严格程度
- 最后获得图片中 表格不同位置的标签，置信度和图像四个点的坐标位置
### 修正表格每行的位置
- 提取行/表头的左右边界坐标
- 计算统一的左右边界（数量多则取均值，数量少则取极值，避免异常值影响）
- 修正行/表头的左右边界，确保统一对齐
- 提取列的上下边界坐标
- 计算统一的上下边界（数量多则取中位数，数量少则取极值）
- 修正列的上下边界，确保统一对齐
### 获得每张表每个表属性对应的表layout框的全局坐标
### 获取表格的不同属性
- layouts_cleanup步骤同版面识别时去除重叠的layout框步骤
```python
headers = gather(r".*header$")
rows = gather(r".* (row|header)")
spans = gather(r".*spanning")
clmns = sorted([r for r in self.tb_cpns if re.match(r"table column$", r["label"])],
               key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
```
### 根据表格属性为文本框添加键值对
```python
for b in self.boxes:
if b.get("layout_type", "") != "table":
    continue
ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)  # 关联文本框到表格行
if ii is not None:
    b["R"] = ii # 标记所属行索引
    b["R_top"] = rows[ii]["top"] # 记录该行的上边界
    b["R_bott"] = rows[ii]["bottom"] # 记录该行的下边界

ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)  # 关联文本框到表头
if ii is not None: 
    b["H_top"] = headers[ii]["top"] # 表头的上边界
    b["H_bott"] = headers[ii]["bottom"] # 表头的下边界
    b["H_left"] = headers[ii]["x0"] # 表头的左边界
    b["H_right"] = headers[ii]["x1"] # 表头的右边界
    b["H"] = ii # 标记所属表头索引

ii = Recognizer.find_horizontally_tightest_fit(b, clmns)  # 关联文本框到表格列
if ii is not None:
    b["C"] = ii # 标记所属列索引
    b["C_left"] = clmns[ii]["x0"]  # 该列的左边界
    b["C_right"] = clmns[ii]["x1"]  # 该列的右边界

ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)  # 关联文本框到合并单元格
if ii is not None:
    b["H_top"] = spans[ii]["top"]  # 合并单元格的上边界
    b["H_bott"] = spans[ii]["bottom"]  # 合并单元格的下边界
    b["H_left"] = spans[ii]["x0"]  # 合并单元格的左边界
    b["H_right"] = spans[ii]["x1"]  # 合并单元格的右边界
    b["SP"] = ii  # 标记所属合并单元格索引
```
## 对所有文本框进行合并文本
### 根据相对于最左坐标的缩进，对每页的表格列按照位置进行分组
- 先获得每个页的文本框
- 获取页面的左右边界 设置可缩进阈值，当文本框左坐标标位置和整个页面最左坐标位置之间的距离小于阈值也认为当前文本框的左坐标为页面最左坐标
- 聚类最多为4个缩进层级，分别遍历取最优结果
- 提取聚类中心（每列的核心左边界）  对聚类中心排序，得到“左→右”的索引顺序
- 标签重映射：初始标签→0/1/2...（左列=0，右列=1）
- 为文本框标记最终列ID
### 合并文本框
- 当页数相同，列数(缩进层级)相同，layoutno相同
- 并且两个文本块(topa+bottoma-topb-bottomb)小于页面平均高度的1/3时
- 合并文本
```python
if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
    # merge
    bxs[i]["x1"] = b_["x1"]
    bxs[i]["top"] = (b["top"] + b_["top"]) / 2
    bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
    bxs[i]["text"] += b_["text"]
    bxs.pop(i + 1)
```
### 提取表格、图片
- 提取表格、图片单独保存，从文本框列表中删除
- 对表格按“顶部坐标（top）→ 左侧坐标（x0）”排序（保证表格按阅读顺序排列）
- 在剩下的文本框列表中检测表格的开头
- 计算表格开头的文本框和表格列表/图标列表中文本框之间的距离
- 选择最近距离，加入到最近距离类型的首位
### 裁剪表格、图片
#### 裁剪图片，根据图片所占页数采取不同策略，裁剪整体图片
- 提取图片元素所属页码判断是否为单页 如果为单页
```python
pn = list(pn)[0]  # 单页页码（索引从0开始）
ht = self.page_cum_height[pn]  # 当前页之前的累计高度（用于坐标转换）

# 步骤2：基于文本框计算元素的粗略边界（取所有文本框的极值）
b = {
    "x0": np.min([b["x0"] for b in bxs]),  # 左边界（所有文本框最小x0）
    "top": np.min([b["top"] for b in bxs]) - ht,  # 上边界（减去累计高度，转为页面内坐标）
    "x1": np.max([b["x1"] for b in bxs]),  # 右边界（所有文本框最大x1）
    "bottom": np.max([b["bottom"] for b in bxs]) - ht  # 下边界（转为页面内坐标）
}

# 步骤3：匹配页面内的精准布局（优先用预解析的布局信息，而非文本框极值）
louts = [layout for layout in self.page_layout[pn] if layout["type"] == ltype]  # 筛选同类型布局
ii = Recognizer.find_overlapped(b, louts, naive=True)  # 找到与粗略边界重叠的精准布局
if ii is not None:
    b = louts[ii]  # 替换为精准布局的边界（更准确）
else:
    logging.warning(f"Missing layout match: {pn + 1},%s" % (bxs[0].get("layoutno", "")))  # 无匹配布局，告警

# 步骤4：提取最终裁剪边界（容错：避免右边界<左边界）
left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
if right < left:
    right = left + 1  # 容错：保证裁剪宽度至少1px

# 步骤5：记录裁剪位置（页码+偏移、左右上下边界）
poss.append((pn + self.page_from, left, right, top, bott))

# 步骤6：裁剪图片（乘以ZM还原为原始像素坐标）
return self.page_images[pn].crop((left * ZM, top * ZM, right * ZM, bott * ZM)
```
- 提取图片元素所属页码 如果不为单页
```python
# 步骤1：按页码分组文本框（跨页元素拆分为单页子元素）
pn = {}
for b in bxs:
    p = b["page_number"] - 1
    if p not in pn:
        pn[p] = []
    pn[p].append(b)
pn = sorted(pn.items(), key=lambda x: x[0])  # 按页码升序排列（保证拼接顺序）

# 步骤2：递归调用cropout，逐页裁剪子元素
imgs = [cropout(arr, ltype, poss) for p, arr in pn]

# 步骤3：创建空白画布，垂直拼接所有单页裁剪图
# 画布宽度：取所有子图的最大宽度；高度：所有子图高度之和；背景色：浅灰色(245,245,245)
pic = Image.new(
    "RGB", 
    (int(np.max([i.size[0] for i in imgs])), int(np.sum([m.size[1] for m in imgs]))),
    (245, 245, 245)
)

# 步骤4：逐页粘贴子图到画布（垂直拼接）
height = 0
for img in imgs:
    pic.paste(img, (0, int(height)))  # 从顶部开始，依次粘贴
    height += img.size[1]

# 步骤5：返回拼接后的完整图片
return pic
```
#### 裁剪表格
- 将表格列表中的文本框按照上下左右排序
-
- 结合乱七八糟的分词，半角转全角，繁体转简体等，确定表格文本框的btype，应该就是确定单元格的文本所属类型
##### 按行进行分组
- 对单元格进行排序 先按行标识（R）排序，再按垂直位置（top/bottom）排序
- 根据前后文本框的行标识（R）不同 → 属于不同行；当前文本框的top ≥ 基准下边界-3，且行标识不连续 → 属于新行（3为像素容差）；
- 把文本框按照行进行分类
##### 按列进行分组
- 判断是否为跨页表格（文本框分布在多个页码）
- 跨页表格：按水平坐标（X）优先排序（避免跨页列乱序）
- 非跨页表格：按列标识（C）优先排序（更精准）
- 列标识连续（当前C - 上一个C = 1）且同页码 → 是新列；
- 当前文本框左边界 ≥ 基准右边界（水平距离足够远）且列标识不异常 → 是新列；
##### 记录当前文本框的行数和列数 形成二维坐标
##### 对于大于4行的表格进行额外处理
- 合并孤立列，合并孤立行 根据距离判断前合并还是后合并
##### 判断表头
- 单元格内任意文本框有表头标记H=True
- 表格主导类型是数值，但当前单元格非数值
##### 将表格转化为html
- 计算列 / 行的平均边界（为合并判断做基准）
- 遍历文本框，计算跨行 / 跨列索引列表
- 遍历二维表格tbl，处理每个单元格
- 单元格内无合并标记（rowspan/colspan），跳过
- 收集单元格内所有文本框的跨行/跨列索引
- 去重并排序
- 无有效合并（跨行/跨列数<2）：删除合并标记，视为普通单元格
- 有有效合并：整理为连续的行/列范围
- 生成连续范围（比如[0,2] → [0,1,2]）
- 合并范围内的单元格内容整合
- 为合并后的文本框设置最终的rowspan/colspan数值
- 合并后的内容只保留在合并范围的首个单元格
## 再次合并文本
-  为文本框分配列ID（标记所属列）
- 按「页码+列ID」分组（仅合并同页同列的文本框）
- 按垂直位置（top）+水平位置（x0）排序（保证从上到下、从左到右遍历） 计算文本框平均高度（作为垂直间距判断的基准）
### 遍历文本框，判断是否垂直合并
- 跨页且当前文本仅含数字/符号 → 移除当前文本框（无效页码/标记）
- 当前文本框无有效文本 → 移除
- 当前文本无内容 或 布局编号不同 → 跳过（不同布局块不合并）
- 垂直间距超过平均高度1.5倍 → 跳过（间距太大，非同一行拆分）
- 水平重叠率<30% → 跳过（水平错位，非同一列文本）
### 文本特征判断
#### 「应该合并」的特征（前一个文本未结束，需和后一个合并）
- 前文本以“未结束标点”结尾
- 倒数第二个字符是未结束标点（容错）
- 后文本以接续标点开头
#### 「不应该合并」的特征（前一个文本已结束，无需合并）
- 布局编号不同（冗余校验）
- 前文本以“结束标点”结尾
- 英文文本以结束标点结尾
- 同页且垂直间距超过平均高度1.5倍（冗余校验）
- 跨页且水平偏移过大（超过平均宽度4倍）
#### 「强制不合并」的位置特征（水平完全分离）
#### 合并判断：满足“不合并特征且无合并特征” 或 “强制不合并” → 跳过
#### 执行文本框合并
- 合并文本：前文本去尾空格 + 空格 + 后文本去头空格，最终去首尾空格
- 更新合并后文本框的边界
- 删除后一个文本框（已合并到当前文本框）
- 当前分组的合并结果加入总列表
- 最终排序：按「页码+列ID+垂直位置」排序，保证文本框顺序符合阅读逻辑
## 移除文档中「目录页、致谢页」的无效文本
### 过滤目录 / 致谢类文本（优先级更高）
- 匹配“目录/致谢”类文本（中英文，忽略空格/全角空格）
- 判断是否为英文目录（文本含5个以上英文字母/数字/符号）
- 提取目录项前缀（用于定位后续目录文本）
- 前缀为空则继续删除文本框，直到找到有效前缀
- 删除前缀所在的文本框
- 批量删除目录区域文本（最多检查后续128个文本框）
### 过滤含大量分隔符的脏页面文本
- 统计每个页面含“··”分隔符的文本框数
- 标记脏页：含分隔符文本框数>3的页面
- 删除脏页中的所有文本框
## 基于项目符号 / 前缀字符” 合并文本框
- 当前文本框无有效文本 → 移除，不合并
- 下一个文本框无有效文本 → 移除，不合并
### 核心判断：不满足合并条件 → 跳过
- 首字符不同（非同一项目符号）
- 首字符是英文字母（避免普通英文单词误合并）
- 首字符是中文（避免普通中文文本误合并）
- 当前文本框在上、下一个在下（垂直位置不重叠/相邻）
- 执行合并：将当前文本框内容合并到下一个文本框
```python
b_["text"] = b["text"] + "\n" + b_["text"]  # 保留换行，还原原始排版
b_["x0"] = min(b["x0"], b_["x0"])  # 合并后左边界取最小值（覆盖更宽）
b_["x1"] = max(b["x1"], b_["x1"])  # 合并后右边界取最大值（覆盖更宽）
b_["top"] = b["top"]  # 合并后上边界取当前文本框的top（还原顶部位置）
self.boxes.pop(i)  # 删除当前文本框（已合并到下一个）
```