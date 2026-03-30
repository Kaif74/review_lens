from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag


RATING_RE = re.compile(r"(\d(?:\.\d+)?)\s*(?:out of|/)\s*(\d(?:\.\d+)?)", re.IGNORECASE)
STAR_RE = re.compile(r"(\d(?:\.\d+)?)\s*star", re.IGNORECASE)
DATE_RE = re.compile(
    r"((?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:\d{1,2}\s+[A-Za-z]+\s+\d{4})|(?:[A-Za-z]+\s+\d{1,2},\s+\d{4}))"
)
REVIEW_LINK_TEXT_RE = re.compile(r"\b(review|ratings?)\b", re.IGNORECASE)
REVIEW_LINK_HREF_RE = re.compile(r"(review|ratings?)", re.IGNORECASE)


def extract_product_name(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    meta = soup.select_one('meta[property="og:title"]') or soup.select_one('meta[name="twitter:title"]')
    if meta and meta.get("content"):
        return str(meta["content"]).strip()

    for obj in _json_ld_objects(soup):
        item_types = _normalize_types(obj.get("@type"))
        if "Product" in item_types:
            name = obj.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    heading = soup.find(["h1", "title"])
    if heading:
        return heading.get_text(" ", strip=True) or None
    return None


def extract_reviews(html: str, source_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    product_name = extract_product_name(html)
    host = urlparse(source_url).netloc.lower()

    site_reviews: list[dict[str, Any]] = []
    if "amazon." in host:
        site_reviews = _extract_amazon_reviews(soup, source_url, product_name)
    elif "flipkart.com" in host:
        site_reviews = _extract_flipkart_reviews(soup, source_url, product_name)
    elif "bestbuy.com" in host:
        site_reviews = _extract_bestbuy_reviews(soup, source_url, product_name)

    if site_reviews:
        return _dedupe_reviews(site_reviews)

    reviews = _extract_json_ld_reviews(soup, source_url, product_name)
    if reviews:
        return _dedupe_reviews(reviews)

    reviews = _extract_html_reviews(soup, source_url, product_name)
    return _dedupe_reviews(reviews)


def discover_review_links(html: str, source_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    parsed_source = urlparse(source_url)
    source_path = parsed_source.path.rstrip("/").lower()
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue

        absolute_url = urljoin(source_url, href)
        parsed = urlparse(absolute_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc and parsed_source.netloc and parsed.netloc != parsed_source.netloc:
            continue

        normalized = absolute_url.split("#", 1)[0]
        if normalized in seen:
            continue

        text = _clean_text(anchor.get_text(" ", strip=True))
        href_text = f"{href} {parsed.path}".strip()
        if not REVIEW_LINK_TEXT_RE.search(text) and not REVIEW_LINK_HREF_RE.search(href_text):
            continue

        score = 0
        if "show all reviews" in text.lower():
            score += 6
        if "customer reviews" in text.lower():
            score += 5
        if "ratings & reviews" in text.lower() or "ratings and reviews" in text.lower():
            score += 5
        if "/reviews" in parsed.path.lower():
            score += 4
        if "review" in parsed.path.lower():
            score += 3
        if parsed.path.rstrip("/").lower() != source_path:
            score += 2
        if parsed.query:
            score += 1

        candidates.append((score, normalized))
        seen.add(normalized)

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, url in candidates[:5]]


def _extract_json_ld_reviews(
    soup: BeautifulSoup,
    source_url: str,
    product_name: str | None,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for obj in _json_ld_objects(soup):
        item_types = _normalize_types(obj.get("@type"))

        if "Review" in item_types:
            review = _review_from_json_ld(obj, source_url, product_name)
            if review:
                reviews.append(review)

        if "Product" in item_types:
            nested_reviews = obj.get("review")
            if isinstance(nested_reviews, dict):
                nested_reviews = [nested_reviews]
            if isinstance(nested_reviews, list):
                for item in nested_reviews:
                    if isinstance(item, dict):
                        review = _review_from_json_ld(item, source_url, product_name)
                        if review:
                            reviews.append(review)

        if "AggregateRating" in item_types:
            aggregate = _aggregate_from_json_ld(obj, source_url, product_name)
            if aggregate:
                reviews.append(aggregate)

    return reviews


def _extract_amazon_reviews(
    soup: BeautifulSoup,
    source_url: str,
    product_name: str | None,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for node in soup.select('[data-hook="review"]'):
        body_node = node.select_one('[data-hook="review-body"]') or node.select_one('[data-hook="review-collapsed"]')
        title_node = node.select_one('[data-hook="review-title"]')
        rating_node = node.select_one('[data-hook="review-star-rating"]') or node.select_one('[data-hook="cmps-review-star-rating"]')
        author_node = node.select_one(".a-profile-name")
        date_node = node.select_one('[data-hook="review-date"]')

        body = _clean_text(body_node.get_text(" ", strip=True) if body_node else "")
        if len(body) < 10:
            continue

        reviews.append(
            {
                "source_url": source_url,
                "page_url": source_url,
                "product_name": product_name,
                "author": _clean_text(author_node.get_text(" ", strip=True) if author_node else "") or "Anonymous",
                "rating": _parse_rating(rating_node.get_text(" ", strip=True) if rating_node else None),
                "review_date": _parse_date(date_node.get_text(" ", strip=True) if date_node else None),
                "title": _clean_text(title_node.get_text(" ", strip=True) if title_node else None) or None,
                "review_text": body,
                "raw_metadata": {
                    "source": "amazon-html",
                    "verified_purchase": node.select_one('[data-hook="avp-badge"]') is not None,
                },
            }
        )
    return reviews


def _extract_flipkart_reviews(
    soup: BeautifulSoup,
    source_url: str,
    product_name: str | None,
) -> list[dict[str, Any]]:
    block_reviews = _extract_flipkart_review_blocks(soup, source_url, product_name)
    if block_reviews:
        return block_reviews

    reviews: list[dict[str, Any]] = []
    candidates: list[Tag] = []
    for node in soup.find_all(["div", "article", "section", "li"]):
        if not isinstance(node, Tag):
            continue
        text = _clean_text(node.get_text(" ", strip=True))
        lowered = text.lower()
        has_buyer_marker = "certified buyer" in lowered or "flipkart customer" in lowered or "verified buyer" in lowered
        has_review_text = node.find("p") is not None or node.select_one("div._6K-7Co, p.z9E0IG, div._11pzQk") is not None
        if has_buyer_marker and has_review_text:
            candidates.append(node)

    for node in candidates:
        text = _clean_text(node.get_text(" ", strip=True))
        rating = _rating_from_text(text) or _parse_rating(_first_matching_text(node, [r"\b[1-5](?:\.\d)?\s*★", r"\b[1-5](?:\.\d)?\b"]))
        body = _clean_flipkart_body(_extract_review_text_from_container(node, text))
        if not body or len(body) < 10:
            continue
        author = _extract_flipkart_author(text) or _extract_author(node) or "Anonymous"
        date = _extract_flipkart_date(text) or _extract_date(node)
        title = _extract_flipkart_title(node, body)
        reviews.append(
            {
                "source_url": source_url,
                "page_url": source_url,
                "product_name": product_name,
                "author": author,
                "rating": rating,
                "review_date": date,
                "title": title,
                "review_text": body,
                "raw_metadata": {
                    "source": "flipkart-html",
                    "verified_purchase": any(marker in text.lower() for marker in ["certified buyer", "verified buyer"]),
                },
            }
        )
    return reviews


def _extract_flipkart_review_blocks(
    soup: BeautifulSoup,
    source_url: str,
    product_name: str | None,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    selectors = [
        "div._16PBlm",
        "div.col.EPCmJX",
        "div.row._3wYu6I",
        "div._27M-vq",
    ]
    nodes: list[Tag] = []
    seen_nodes: set[int] = set()
    for selector in selectors:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue
            marker = id(node)
            if marker in seen_nodes:
                continue
            seen_nodes.add(marker)
            nodes.append(node)

    for node in nodes:
        text = _clean_text(node.get_text(" ", strip=True))
        rating_node = (
            node.select_one("div._3LWZlK")
            or node.select_one("div._3LWZlK._1BLPMq")
            or node.select_one("div.XQDdHH")
        )
        title_node = (
            node.select_one("p._2-N8zT")
            or node.select_one("p._2xg6Ul")
            or node.select_one("div._6K-7Co")
            or node.select_one("p.z9E0IG")
        )
        body_node = (
            node.select_one("div.t-ZTKy")
            or node.select_one("div.t-ZTKy > div")
            or node.select_one("div.ZmyHeo")
            or node.select_one("div._11pzQk")
        )
        author_node = (
            node.select_one("p._2sc7ZR")
            or node.select_one("div._2sc7ZR")
            or node.select_one("span._2sc7ZR")
        )
        date_node = node.find(string=re.compile(r"\b(?:day|days|month|months|year|years)\s+ago\b", re.IGNORECASE))

        body = _clean_flipkart_body(_clean_text(body_node.get_text(" ", strip=True) if body_node else ""))
        if not body or len(body) < 5:
            continue

        reviews.append(
            {
                "source_url": source_url,
                "page_url": source_url,
                "product_name": product_name,
                "author": _clean_text(author_node.get_text(" ", strip=True) if author_node else "") or _extract_flipkart_author(text) or "Anonymous",
                "rating": _parse_rating(rating_node.get_text(" ", strip=True) if rating_node else None),
                "review_date": _parse_date(date_node) if date_node else _extract_flipkart_date(text) or _extract_date(node),
                "title": _clean_text(title_node.get_text(" ", strip=True) if title_node else None) or None,
                "review_text": body,
                "raw_metadata": {
                    "source": "flipkart-review-block",
                    "verified_purchase": any(marker in text.lower() for marker in ["certified buyer", "verified buyer"]),
                },
            }
        )

    return _dedupe_reviews(reviews)


def _extract_bestbuy_reviews(
    soup: BeautifulSoup,
    source_url: str,
    product_name: str | None,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    selectors = [
        "article.review-item",
        "li.review-item",
        "[data-testid='review-item']",
        "[data-track='customer-review']",
    ]

    nodes: list[Tag] = []
    for selector in selectors:
        nodes.extend(node for node in soup.select(selector) if isinstance(node, Tag))

    if not nodes:
        for node in soup.find_all(["article", "li", "div", "section"]):
            if not isinstance(node, Tag):
                continue
            text = _clean_text(node.get_text(" ", strip=True))
            lowered = text.lower()
            if "posted" in lowered and ("user rating" in lowered or "out of 5" in lowered or "star" in lowered):
                nodes.append(node)

    for node in nodes:
        text = _clean_text(node.get_text(" ", strip=True))
        body = _extract_review_text_from_container(node, text)
        if not body or len(body) < 10:
            continue
        title_node = node.find(["h3", "h4", "strong"])
        reviews.append(
            {
                "source_url": source_url,
                "page_url": source_url,
                "product_name": product_name,
                "author": _extract_author(node) or "Anonymous",
                "rating": _rating_from_text(text),
                "review_date": _extract_bestbuy_date(text) or _extract_date(node),
                "title": _clean_text(title_node.get_text(" ", strip=True) if title_node else None) or None,
                "review_text": body,
                "raw_metadata": {
                    "source": "bestbuy-html",
                    "verified_purchase": "verified purchase" in text.lower(),
                },
            }
        )
    return reviews


def _review_from_json_ld(
    obj: dict[str, Any],
    source_url: str,
    product_name: str | None,
) -> dict[str, Any] | None:
    text = obj.get("reviewBody") or obj.get("description")
    if not isinstance(text, str) or not text.strip():
        return None

    author = obj.get("author")
    if isinstance(author, dict):
        author = author.get("name")

    return {
        "source_url": source_url,
        "page_url": source_url,
        "product_name": product_name,
        "author": author if isinstance(author, str) and author.strip() else "Anonymous",
        "rating": _parse_rating(obj.get("reviewRating")),
        "review_date": _parse_date(obj.get("datePublished")),
        "title": _clean_text(obj.get("name")) if isinstance(obj.get("name"), str) else None,
        "review_text": _clean_text(text),
        "raw_metadata": {"source": "json-ld"},
    }


def _aggregate_from_json_ld(
    obj: dict[str, Any],
    source_url: str,
    product_name: str | None,
) -> dict[str, Any] | None:
    rating = _parse_rating(obj.get("ratingValue") or obj.get("rating"))
    count = obj.get("reviewCount") or obj.get("ratingCount")
    if rating is None:
        return None

    summary = f"Aggregate rating {rating} out of 5"
    if count is not None:
        summary += f" based on {count} ratings."

    return {
        "source_url": source_url,
        "page_url": source_url,
        "product_name": product_name,
        "author": "AggregateRating",
        "rating": rating,
        "review_date": None,
        "title": "Aggregate rating",
        "review_text": summary,
        "raw_metadata": {"source": "aggregate-rating"},
    }


def _extract_html_reviews(
    soup: BeautifulSoup,
    source_url: str,
    product_name: str | None,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    seen_blocks: set[str] = set()

    for node in soup.find_all(["article", "section", "div", "li"]):
        if not isinstance(node, Tag):
            continue
        text = node.get_text(" ", strip=True)
        cleaned_text = _clean_text(text)
        if len(cleaned_text) < 40 or cleaned_text in seen_blocks:
            continue

        rating = _rating_from_text(cleaned_text)
        if rating is None:
            continue

        review_text = _extract_review_text_from_container(node, cleaned_text)
        if not review_text or len(review_text) < 20:
            continue

        seen_blocks.add(cleaned_text)
        reviews.append(
            {
                "source_url": source_url,
                "page_url": source_url,
                "product_name": product_name,
                "author": _extract_author(node) or "Anonymous",
                "rating": rating,
                "review_date": _extract_date(node),
                "title": None,
                "review_text": review_text,
                "raw_metadata": {"source": "html-heuristic"},
            }
        )

    return reviews


def _extract_review_text_from_container(node: Tag, fallback_text: str) -> str | None:
    for selector in ["p", "blockquote", "span", "div"]:
        for child in node.select(selector):
            text = _clean_text(child.get_text(" ", strip=True))
            if len(text) >= 20 and text != fallback_text and _rating_from_text(text) is None:
                return text

    text = re.sub(RATING_RE, "", fallback_text)
    text = re.sub(STAR_RE, "", text)
    return _clean_text(text)


def _extract_flipkart_author(text: str) -> str | None:
    name_pattern = r"([A-Z][A-Za-z._-]+(?:\s+[A-Z][A-Za-z._-]+){0,3})"
    match = re.search(name_pattern + r"\s+Certified Buyer", text)
    if match:
        return _clean_text(match.group(1))
    match = re.search(name_pattern + r"\s+Verified Buyer", text)
    if match:
        return _clean_text(match.group(1))
    match = re.search(name_pattern + r"\s+Flipkart Customer", text)
    if match:
        return _clean_text(match.group(1))
    return None


def _extract_flipkart_date(text: str) -> str | None:
    match = re.search(r"(\d+\s+(?:day|days|month|months|year|years)\s+ago)", text, re.IGNORECASE)
    if match:
        return _clean_text(match.group(1))
    return None


def _extract_flipkart_title(node: Tag, body: str) -> str | None:
    selectors = ["div._6K-7Co", "p.z9E0IG", "div._11pzQk", "span", "div", "p"]
    for selector in selectors:
        for child in node.select(selector):
            text = _clean_text(child.get_text(" ", strip=True))
            if (
                5 <= len(text) <= 120
                and text != body
                and _clean_flipkart_body(text) != body
                and len(text.split()) <= 12
                and "verified buyer" not in text.lower()
                and "certified buyer" not in text.lower()
            ):
                return text
    return None


def _extract_bestbuy_date(text: str) -> str | None:
    match = re.search(r"Posted\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if match:
        return _clean_text(match.group(1))
    return None


def _extract_author(node: Tag) -> str | None:
    selectors = [
        '[itemprop="author"]',
        '[class*="author"]',
        '[class*="user"]',
        '[class*="profile"]',
    ]
    for selector in selectors:
        match = node.select_one(selector)
        if match:
            text = _clean_text(match.get_text(" ", strip=True))
            if text and len(text) <= 80:
                return text

    text = _clean_text(node.get_text(" ", strip=True))
    by_match = re.search(r"\bby\s+([A-Z][A-Za-z0-9 ._-]{1,40})", text)
    if by_match:
        return by_match.group(1).strip()
    return None


def _extract_date(node: Tag) -> str | None:
    time_node = node.find("time")
    if time_node:
        raw = time_node.get("datetime") or time_node.get_text(" ", strip=True)
        return _parse_date(raw)

    text = _clean_text(node.get_text(" ", strip=True))
    match = DATE_RE.search(text)
    return _parse_date(match.group(1)) if match else None


def _rating_from_text(text: str) -> float | None:
    match = RATING_RE.search(text)
    if match:
        return _parse_rating(match.group(1))
    match = STAR_RE.search(text)
    if match:
        return _parse_rating(match.group(1))
    return None


def _parse_rating(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("ratingValue") or raw.get("value")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if match:
            return float(match.group(1))
    return None


def _parse_date(raw: Any) -> str | None:
    if raw is None:
        return None
    text = _clean_text(str(raw))
    return text or None


def _json_ld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for item in _walk_json(parsed):
            if isinstance(item, dict):
                objects.append(item)
    return objects


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _normalize_types(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None:
        return []
    return [str(raw)]


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_flipkart_body(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = _clean_text(re.sub(r"\bRead more\b", "", text, flags=re.IGNORECASE))
    return cleaned or None


def _first_matching_text(node: Tag, patterns: list[str]) -> str | None:
    haystack = _clean_text(node.get_text(" ", strip=True))
    for pattern in patterns:
        match = re.search(pattern, haystack, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _dedupe_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for review in reviews:
        key = (
            str(review.get("author") or "").strip().lower(),
            str(review.get("review_date") or "").strip().lower(),
            str(review.get("review_text") or "").strip().lower()[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(review)
    return deduped
