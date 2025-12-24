import requests
import random
import time
import os
from pathlib import Path
from urllib.parse import urlencode, urlparse, urljoin, parse_qs, unquote
from bs4 import BeautifulSoup
from readability import Document
from langchain_community.document_loaders import UnstructuredHTMLLoader

BASE_DIR = Path(__file__).parent


class WebSearch:
    def __init__(self):
        self.OUTPUT_DIR = BASE_DIR / "pages"
        self.BLACKLIST = ("login", "signup", "account", "captcha", "auth", "cookie")
        self.MAX_LINKS_PER_PAGE = 20
        self.session = self._get_session()

    @staticmethod
    def _get_session():
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        session = requests.Session()
        session.headers.update(headers)
        return session

    @staticmethod
    def human_delay():
        time.sleep(random.uniform(1.2, 3.5))

    def risky_url(self, url):
        u = url.lower()
        return any(x in u for x in self.BLACKLIST)

    @staticmethod
    def same_domain(url1, url2):
        return urlparse(url1).netloc == urlparse(url2).netloc

    def safe_get(self, url, referer=None, retries=2):
        if self.risky_url(url):
            print("跳过高风险 URL:", url)
            return None

        for _ in range(retries):
            try:
                headers = {"Referer": referer} if referer else {}
                self.human_delay()
                resp = self.session.get(url, headers=headers, timeout=10)
                if resp.status_code == 403:
                    print("403:", url)
                    continue
                resp.raise_for_status()
                return resp
            except Exception as e:
                print("请求失败:", url, e)
                self.human_delay()
        return None

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
            real_url = unquote(qs["uddg"][0])
            return real_url
        return ddg_url

    def extract_ddg_links(self, html):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", class_="result__a", href=True):
            real = self.extract_real_url(a["href"])
            links.append(real)
        return links

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

    def baidu_search(self, query, limit=10):
        url = "https://www.baidu.com/s"
        params = {"wd": query}

        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print("百度搜索失败:", e)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
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

    def resolve_baidu_url(self, baidu_url):
        try:
            resp = self.session.get(
                baidu_url,
                allow_redirects=True,
                timeout=10
            )
            return resp.url
        except Exception:
            return None

    def search_urls(self, query):
        """
        优先 百度，失败则DDG
        """
        real_urls = []
        try:
            baidu_links = self.baidu_search(query)

            for link in baidu_links:
                real = self.resolve_baidu_url(link)
                if real:
                    real_urls.append(real)
            if real_urls:
                return real_urls
        except:
            pass

        # ===== 2. 尝试 DuckDuckGo =====
        print("尝试 DuckDuckGo")

        try:
            params = urlencode({"q": query})
            search_url = f"https://duckduckgo.com/html/?{params}"
            resp = self.safe_get(search_url)
            if resp:
                urls = self.extract_ddg_links(resp.text)
                if urls:
                    print("使用 DuckDuckGo 搜索")
                    return urls
        except Exception:
            pass

        return real_urls

    def crawl_search_query(self, query, max_results=5):
        # params = urlencode({"q": query})
        # search_url = f"https://duckduckgo.com/html/?{params}"
        #
        # resp = self.safe_get(search_url)
        # if not resp:
        #     print("搜索页请求失败")
        #     return []
        #
        # urls = self.extract_ddg_links(resp.text)
        # print("搜索结果数量:", len(urls))
        # leaf_pages = []
        urls = self.search_urls(query)

        if not urls:
            print("搜索失败，无可用结果")
            return []

        leaf_pages = []
        for url in urls:
            html_resp = self.safe_get(url)
            if not html_resp:
                continue

            html = html_resp.text
            if self.is_leaf_page(html):
                print("找到叶子页:", url)
                leaf_pages.append((url, html))
            else:
                # 继续抓取同域名子链接
                retry_count = 0
                for link in self.extract_links(url, html):
                    if retry_count > 2:
                        break
                    if self.same_domain(url, link):
                        sub_resp = self.safe_get(link, referer=url)
                        if sub_resp and self.is_leaf_page(sub_resp.text):
                            leaf_pages.append((link, sub_resp.text))
                        retry_count += 1
            if len(leaf_pages) >= max_results:
                break
        return leaf_pages

    def save_leaf_page(self, url: str, html: str) -> str:
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        doc = Document(html)
        content_html = doc.summary(html_partial=True)

        parsed = urlparse(url)
        safe_name = (
                parsed.netloc.replace(".", "_")
                + parsed.path.replace("/", "_")[:50]
        )

        path = os.path.join(self.OUTPUT_DIR, safe_name + ".html")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content_html)

        return path

    def __call__(self, query, limit=5):
        pages = self.crawl_search_query(query, limit)
        print(f"\n抓取到叶子页数量: {len(pages)}")
        htmls_content = []
        for url, html in pages:
            path = self.save_leaf_page(url, html)
            print("Saved:", path)

            loader = UnstructuredHTMLLoader(path)
            docs = loader.load()
            html_content = docs[0].page_content
            htmls_content.append(html_content)
            os.remove(path)
            print("Parsed docs:", html_content)
        if not htmls_content:
            return ''
        return '----'.join(htmls_content)


if __name__ == '__main__':
    query = 'python'
    search = WebSearch()
    print(search(query))
