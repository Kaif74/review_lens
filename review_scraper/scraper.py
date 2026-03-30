from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .browser import BrowserFetchError, BrowserFetcher
from .models import ScraperConfig


class ScrapeError(RuntimeError):
    """Raised when the application cannot fetch a page."""


BESTBUY_PRODUCT_WITH_SKU_RE = re.compile(r"^/product/(?P<slug>.+?)(?:/[^/]+)?/sku/(?P<sku>[A-Za-z0-9]+)/?$", re.IGNORECASE)
BESTBUY_PRODUCT_ID_RE = re.compile(r"^/product/(?P<slug>.+?)/(?P<product_id>[A-Za-z0-9]{6,})(?:/)?$", re.IGNORECASE)
BESTBUY_SITE_RE = re.compile(r"^/site/(?P<slug>.+?)/(?P<product_id>[A-Za-z0-9]+)(?:\.p)?/?$", re.IGNORECASE)
AMAZON_ASIN_RE = re.compile(r"/(?:dp|gp/product|product-reviews)/(?P<asin>[A-Z0-9]{10})", re.IGNORECASE)
FLIPKART_ITEM_RE = re.compile(r"/(?:p|product-reviews)/(?P<item>itm[a-z0-9]+)", re.IGNORECASE)
FALLBACK_HTTP_STATUSES = {
    401,
    403,
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
    520,
    521,
    522,
    523,
    524,
    529,
}


class ReviewScraper:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        retry_kwargs = {
            "total": max(self.config.max_retries, 0),
            "connect": max(self.config.max_retries, 0),
            "read": max(self.config.max_retries, 0),
            "backoff_factor": 1.0,
            "status_forcelist": tuple(sorted(FALLBACK_HTTP_STATUSES)),
        }
        try:
            retry = Retry(
                **retry_kwargs,
                allowed_methods=frozenset({"GET", "HEAD"}),
            )
        except TypeError:
            retry = Retry(
                **retry_kwargs,
                method_whitelist=frozenset({"GET", "HEAD"}),
            )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        )
        self._browser_fetcher = BrowserFetcher(self.config) if self.config.enable_browser_fallback else None

    def fetch_html(self, url: str) -> str:
        url = normalize_product_url(url)
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else url
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": origin,
        }

        try:
            response = self.session.get(url, headers=headers, timeout=self.config.request_timeout)
        except requests.RequestException as exc:
            return self._fetch_with_browser_or_raise(url, exc)

        if response.status_code != 200:
            if response.status_code in FALLBACK_HTTP_STATUSES:
                return self._fetch_with_browser_or_raise(url, f"HTTP {response.status_code}")
            raise ScrapeError(f"Fetch failed for {url}: HTTP {response.status_code}")

        return response.text

    def scrape(self, url: str) -> str:
        return self.fetch_html(url)

    def build_candidate_urls(self, url: str, max_pages: int = 1) -> list[str]:
        normalized = normalize_product_url(url)
        candidates = build_candidate_urls(normalized, max_pages=max_pages)
        if normalized not in candidates:
            candidates.append(normalized)
        return _dedupe_urls(candidates)

    def _fetch_with_browser_or_raise(self, url: str, original_error) -> str:
        if self._browser_fetcher is None:
            raise ScrapeError(f"Fetch failed for {url}: {original_error}") from (
                original_error if isinstance(original_error, Exception) else None
            )
        try:
            return self._browser_fetcher.fetch_html(url, self.user_agent)
        except BrowserFetchError as exc:
            raise ScrapeError(f"Fetch failed for {url}: {original_error}. {exc}") from exc


def normalize_product_url(url: str) -> str:
    parsed = urlparse(url)
    if "bestbuy.com" not in parsed.netloc.lower():
        return url

    path = parsed.path or ""
    match = BESTBUY_PRODUCT_WITH_SKU_RE.match(path)
    if not match:
        match = BESTBUY_PRODUCT_ID_RE.match(path)
        if not match:
            return url

    slug = match.group("slug").strip("/")
    product_id = match.groupdict().get("sku") or match.groupdict().get("product_id")
    if not slug or not product_id:
        return url

    normalized = f"{parsed.scheme or 'https'}://{parsed.netloc}/site/{slug}/{product_id}.p"
    if product_id.isdigit():
        normalized += f"?skuId={product_id}"
    return normalized


def build_candidate_urls(url: str, max_pages: int = 1) -> list[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    candidates = [url]
    pages = max(1, max_pages)

    if "amazon." in host:
        amazon_urls = _build_amazon_review_urls(parsed, pages)
        if amazon_urls:
            candidates.extend(amazon_urls)
            return _dedupe_urls(candidates)

    if "flipkart.com" in host:
        flipkart_urls = _build_flipkart_review_urls(parsed, pages)
        if flipkart_urls:
            candidates.extend(flipkart_urls)
            return _dedupe_urls(candidates)

    if "bestbuy.com" in host:
        bestbuy_urls = _build_bestbuy_review_urls(parsed, pages)
        if bestbuy_urls:
            candidates.extend(bestbuy_urls)
            return _dedupe_urls(candidates)

    return _dedupe_urls(candidates)


def _build_amazon_review_urls(parsed, max_pages: int) -> list[str]:
    match = AMAZON_ASIN_RE.search(parsed.path or "")
    if not match:
        return []
    asin = match.group("asin").upper()
    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    return [f"{origin}/product-reviews/{asin}/?pageNumber={page}" for page in range(1, max_pages + 1)]


def _build_flipkart_review_urls(parsed, max_pages: int) -> list[str]:
    match = FLIPKART_ITEM_RE.search(parsed.path or "")
    if not match:
        return []
    item = match.group("item")
    pid = parse_qs(parsed.query).get("pid", [None])[0]
    if not pid:
        return []

    path = parsed.path or ""
    if "/product-reviews/" in path:
        base_path = path.split("?", 1)[0]
    else:
        slug_prefix = path.split("/p/")[0].strip("/")
        base_path = f"/{slug_prefix}/product-reviews/{item}" if slug_prefix else f"/product-reviews/{item}"
    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    return [f"{origin}{base_path}?{urlencode({'pid': pid, 'page': page})}" for page in range(1, max_pages + 1)]


def _build_bestbuy_review_urls(parsed, max_pages: int) -> list[str]:
    sku = parse_qs(parsed.query).get("skuId", [None])[0]
    slug = None

    site_match = BESTBUY_SITE_RE.match(parsed.path or "")
    if site_match:
        slug = site_match.group("slug").strip("/")
        product_id = site_match.group("product_id")
        if product_id.isdigit() and not sku:
            sku = product_id

    if not sku:
        product_match = BESTBUY_PRODUCT_WITH_SKU_RE.match(parsed.path or "")
        if product_match:
            slug = product_match.group("slug").strip("/")
            sku = product_match.group("sku")

    if not slug or not sku or not sku.isdigit():
        return []

    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    urls = []
    for page in range(1, max_pages + 1):
        if page == 1:
            urls.append(f"{origin}/site/reviews/{slug}/{sku}")
        else:
            urls.append(f"{origin}/site/reviews/{slug}/{sku}?page={page}")
    return urls


def _dedupe_urls(urls: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped
