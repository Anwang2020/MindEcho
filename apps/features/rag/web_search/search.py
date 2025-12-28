# -*- coding: utf-8 -*-
import asyncio
import random
import time
import os
from pathlib import Path
from urllib.parse import urlencode, urlparse, urljoin, parse_qs, unquote
import threading

import aiohttp
from bs4 import BeautifulSoup
from readability import Document
from langchain_community.document_loaders import UnstructuredHTMLLoader
from apps.logs.logs import get_logger

BASE_DIR = Path(__file__).parent
logger = get_logger(__name__)


class WebSearch:
    """
    异步网页搜索器：按域名限速、自动重试、过滤非HTML，减少频繁请求报错。
    __call__ 为同步入口（内部跑协程）；如在事件循环内，请直接 await asearch。
    """

    def __init__(self):
        self.OUTPUT_DIR = BASE_DIR / "pages"
        self.BLACKLIST = ("login", "signup", "account", "captcha", "auth", "cookie")
        self.MAX_LINKS_PER_PAGE = 20
        self.MIN_INTERVAL_PER_DOMAIN = 1.0
        self.MAX_RETRIES = 3
        self.TIMEOUT = 8
        self.TOTAL_TIMEOUT = 12
        self.GLOBAL_CONCURRENCY = 8
        self.PER_HOST_CONCURRENCY = 2
        self.last_request_ts = {}
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    # ---------- 基础工具 ----------
    async def _respect_rate_limit(self, url: str):
        """按域名限速，避免频繁请求被封。"""
        domain = urlparse(url).netloc
        now = time.monotonic()
        last = self.last_request_ts.get(domain, 0)
        wait = self.MIN_INTERVAL_PER_DOMAIN - (now - last)
        if wait > 0:
            await asyncio.sleep(wait + random.uniform(0.05, 0.2))
        self.last_request_ts[domain] = time.monotonic()

    def risky_url(self, url: str) -> bool:
        u = url.lower()
        return any(x in u for x in self.BLACKLIST)

    @staticmethod
    def same_domain(url1, url2):
        return urlparse(url1).netloc == urlparse(url2).netloc

    # ---------- 网络请求 ----------
    async def safe_get(self, session: aiohttp.ClientSession, url: str, referer: str | None = None,
                       retries: int | None = None) -> str | None:
        if self.risky_url(url):
            logger.info("跳过高风险 URL: %s", url)
            return None

        retries = retries or self.MAX_RETRIES
        for attempt in range(1, retries + 1):
            try:
                await self._respect_rate_limit(url)
                headers = {"Referer": referer} if referer else {}
                async with session.get(url, headers=headers, timeout=self.TIMEOUT) as resp:
                    content_type = resp.headers.get("Content-Type", "")

                    if resp.status in (403, 429, 503):
                        wait = min(6, 0.8 * attempt + random.random())
                        logger.info(f"{resp.status}，退避 {wait:.1f}s:", url)
                        await asyncio.sleep(wait)
                        continue

                    if resp.status >= 400:
                        logger.info("请求失败: %s %s", url, resp.status)
                        continue

                    if "text/html" not in content_type:
                        logger.info("非HTML，跳过: %s %s", url, content_type)
                        return None

                    return await resp.text()
            except Exception as e:
                logger.info("请求异常: %s %s", url, e)
                if attempt < retries:
                    backoff = min(6, 0.5 * attempt + random.uniform(0, 0.5))
                    await asyncio.sleep(backoff)
        return None

    # ---------- 搜索与解析 ----------
    @staticmethod
    def is_leaf_page(html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text_len = sum(len(p.get_text(strip=True)) for p in paragraphs)

        if text_len < 800:
            return False
        if soup.find("article"):
            return True
        return len(paragraphs) > 10

    @staticmethod
    def extract_real_url(ddg_url):
        if ddg_url.startswith("//"):
            ddg_url = "https:" + ddg_url
        parsed = urlparse(ddg_url)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
        return ddg_url

    def extract_ddg_links(self, html):
        soup = BeautifulSoup(html, "html.parser")
        return [self.extract_real_url(a["href"]) for a in soup.find_all("a", class_="result__a", href=True)]

    def extract_links(self, base_url, html):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#"):
                continue
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.scheme in ("http", "https"):
                links.append(full)
            if len(links) >= self.MAX_LINKS_PER_PAGE:
                break
        return links

    async def baidu_search(self, session, query, limit=10):
        params = {"wd": query}
        url = f"https://www.baidu.com/s?{urlencode(params)}"
        html = await self.safe_get(session, url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        urls = []
        for item in soup.select("div.result"):
            if len(urls) >= limit:
                break
            a = item.select_one("h3 a")
            if not a:
                continue
            href = a.get("href")
            if href:
                urls.append(href)
        return urls

    async def resolve_baidu_url(self, session, baidu_url):
        try:
            await self._respect_rate_limit(baidu_url)
            async with session.get(baidu_url, allow_redirects=True, timeout=self.TIMEOUT) as resp:
                return str(resp.url)
        except Exception:
            return None

    async def search_urls(self, session, query):
        """
        优先 百度，失败则 DuckDuckGo
        """
        real_urls = []
        try:
            baidu_links = await self.baidu_search(session, query)
            for link in baidu_links:
                real = await self.resolve_baidu_url(session, link)
                if real and real not in real_urls:
                    real_urls.append(real)
            if real_urls:
                return real_urls
        except Exception:
            pass

        # 尝试 DuckDuckGo
        logger.info("尝试 DuckDuckGo")
        try:
            params = urlencode({"q": query})
            search_url = f"https://duckduckgo.com/html/?{params}"
            html = await self.safe_get(session, search_url)
            if html:
                urls = self.extract_ddg_links(html)
                if urls:
                    logger.info("使用 DuckDuckGo 搜索")
                    return urls
        except Exception:
            pass

        return real_urls

    async def crawl_search_query(self, session, query, max_results=5):
        urls = await self.search_urls(session, query)
        if not urls:
            logger.info("搜索失败，无可用结果")
            return []

        leaf_pages = []
        visited = set()
        for url in urls:
            if url in visited:
                continue
            html = await self.safe_get(session, url)
            if not html:
                continue

            visited.add(url)
            if self.is_leaf_page(html):
                logger.info("找到叶子页: %s", url)
                leaf_pages.append((url, html))
            else:
                # 继续抓取同域名子链接
                retry_count = 0
                for link in self.extract_links(url, html):
                    if retry_count > 2:
                        break
                    if self.same_domain(url, link):
                        if link in visited:
                            continue
                        sub_html = await self.safe_get(session, link, referer=url)
                        if sub_html and self.is_leaf_page(sub_html):
                            leaf_pages.append((link, sub_html))
                        visited.add(link)
                        retry_count += 1
            if len(leaf_pages) >= max_results:
                break
        return leaf_pages

    # ---------- 存储与入口 ----------
    def save_leaf_page(self, url: str, html: str) -> str:
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        doc = Document(html)
        content_html = doc.summary(html_partial=True)

        parsed = urlparse(url)
        safe_name = parsed.netloc.replace(".", "_") + parsed.path.replace("/", "_")[:50]
        path = os.path.join(self.OUTPUT_DIR, safe_name + ".html")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content_html)

        return path

    async def asearch(self, query, limit=5):
        connector = aiohttp.TCPConnector(limit=self.GLOBAL_CONCURRENCY, limit_per_host=self.PER_HOST_CONCURRENCY)
        timeout = aiohttp.ClientTimeout(total=self.TOTAL_TIMEOUT)
        async with aiohttp.ClientSession(headers=self.headers, connector=connector, timeout=timeout) as session:
            pages = await self.crawl_search_query(session, query, limit)
            logger.info(f"\n抓取到叶子页数量: {len(pages)}")
            htmls_content = []
            for url, html in pages:
                path = self.save_leaf_page(url, html)
                logger.info("Saved: %s", path)

                loader = UnstructuredHTMLLoader(path)
                docs = loader.load()
                html_content = docs[0].page_content
                htmls_content.append(html_content)
                os.remove(path)
                logger.info("Parsed docs: %s", html_content)
            if not htmls_content:
                return ''
            return '----'.join(htmls_content)

    def __call__(self, query, limit=5):
        """
        同步入口；如在运行事件循环中，请直接 await self.asearch(query, limit)。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asearch(query, limit))
        else:
            # 如果已经在事件循环内，使用单独线程的事件循环执行以返回同步结果
            result_box = {}

            def runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_box["value"] = loop.run_until_complete(self.asearch(query, limit))
                finally:
                    loop.close()

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            thread.join()
            return result_box.get("value", '')


if __name__ == '__main__':
    query = 'python'
    search = WebSearch()
    result = search(query)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    logger.info(result)
