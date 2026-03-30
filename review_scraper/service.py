from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .adapters import discover_review_links, extract_product_name, extract_reviews
from .llm import ReviewSummarizer
from .models import LLMConfig, ProcessingConfig, ReviewRecord, ScraperConfig
from .preprocess import chunk_text, clean_review_text
from .scraper import ReviewScraper, ScrapeError


class PipelineValidationError(ValueError):
    """Raised when the request payload is invalid."""


@dataclass
class PipelineRequest:
    url: str
    request_timeout: int = 30
    max_pages: int = 1
    max_input_tokens: int = 1200
    chunk_overlap_tokens: int = 80
    tokenizer_model: str = "gpt-4o-mini"
    skip_llm: bool = False
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 400
    llm_rpm: int = 20
    llm_retries: int = 4
    extra_body_json: str | None = None


@dataclass
class PipelineResult:
    product_name: str | None
    reviews: list[ReviewRecord]
    overall_summary: str | None = None
    overall_sentiment: str | None = None
    overall_key_points: list[str] | None = None
    overall_error: str | None = None


ProgressCallback = Callable[[dict[str, Any]], None]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "deepseek-ai/deepseek-v3.1"
NVIDIA_DEPRECATED_MODEL_ALIASES = {
    "deepseek-ai/deepseek-r1",
}
SCRAPE_CACHE_TTL_SECONDS = 15 * 60

_SCRAPE_CACHE: dict[str, dict[str, Any]] = {}
_SCRAPE_CACHE_LOCK = threading.Lock()


def get_runtime_defaults() -> dict[str, Any]:
    defaults = {
        "max_pages": 1,
        "request_timeout": _coerce_int(os.getenv("REVIEW_SCRAPER_REQUEST_TIMEOUT"), default=30),
        "tokenizer_model": os.getenv("REVIEW_SCRAPER_TOKENIZER_MODEL", "gpt-4o-mini"),
        "api_key_env": os.getenv("REVIEW_SCRAPER_API_KEY_ENV"),
        "model": os.getenv("REVIEW_SCRAPER_MODEL"),
        "base_url": os.getenv("REVIEW_SCRAPER_BASE_URL"),
        "extra_body_json": os.getenv("REVIEW_SCRAPER_EXTRA_BODY_JSON") or "",
    }

    if os.getenv("NVIDIA_API_KEY"):
        defaults.update(
            {
                "api_key_env": defaults["api_key_env"] or "NVIDIA_API_KEY",
                "model": defaults["model"] or NVIDIA_DEFAULT_MODEL,
                "base_url": defaults["base_url"] or NVIDIA_BASE_URL,
            }
        )
    elif os.getenv("OPENAI_API_KEY"):
        defaults.update(
            {
                "api_key_env": defaults["api_key_env"] or "OPENAI_API_KEY",
                "model": defaults["model"] or "gpt-4o-mini",
                "base_url": defaults["base_url"] or "",
            }
        )
    else:
        defaults.update(
            {
                "api_key_env": defaults["api_key_env"] or "OPENAI_API_KEY",
                "model": defaults["model"] or "gpt-4o-mini",
                "base_url": defaults["base_url"] or "",
            }
        )

    defaults["model"] = _normalize_model_choice(
        model=defaults["model"],
        base_url=defaults["base_url"],
        api_key_env=defaults["api_key_env"],
    )

    return defaults


def parse_pipeline_request(payload: dict[str, Any]) -> PipelineRequest:
    defaults = get_runtime_defaults()
    url = str(payload.get("url", "")).strip()
    if not url:
        raise PipelineValidationError("A product URL is required.")

    base_url = _coerce_optional_str(payload.get("base_url")) or defaults["base_url"] or None
    api_key_env = str(payload.get("api_key_env") or defaults["api_key_env"]).strip()
    model = _normalize_model_choice(
        model=str(payload.get("model") or defaults["model"]).strip(),
        base_url=base_url,
        api_key_env=api_key_env,
    )

    return PipelineRequest(
        url=url,
        request_timeout=_coerce_int(payload.get("request_timeout"), default=defaults["request_timeout"]),
        max_pages=_coerce_int(payload.get("max_pages"), default=defaults["max_pages"]),
        max_input_tokens=_coerce_int(payload.get("max_input_tokens"), default=1200),
        chunk_overlap_tokens=_coerce_int(payload.get("chunk_overlap_tokens"), default=80),
        tokenizer_model=str(payload.get("tokenizer_model") or defaults["tokenizer_model"]).strip(),
        skip_llm=_coerce_bool(payload.get("skip_llm"), default=False),
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=_coerce_float(payload.get("temperature"), default=0.2),
        top_p=_coerce_float(payload.get("top_p"), default=1.0),
        max_output_tokens=_coerce_int(payload.get("max_output_tokens"), default=400),
        llm_rpm=_coerce_int(payload.get("llm_rpm"), default=20),
        llm_retries=_coerce_int(payload.get("llm_retries"), default=4),
        extra_body_json=_coerce_optional_str(payload.get("extra_body_json")) or defaults["extra_body_json"] or None,
    )


def run_review_pipeline(
    request: PipelineRequest,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    processing_config = ProcessingConfig(
        tokenizer_model=request.tokenizer_model,
        max_input_tokens=request.max_input_tokens,
        chunk_overlap_tokens=request.chunk_overlap_tokens,
    )
    product_name: str | None = None
    review_records: list[ReviewRecord] = []
    fetch_errors: list[str] = []
    seen_review_keys: set[tuple[str | None, str | None, str]] = set()

    _emit_progress(progress_callback, stage="fetching", message="Resolving review pages", progress=5)
    scraper = ReviewScraper(ScraperConfig(request_timeout=request.request_timeout))
    candidate_urls = scraper.build_candidate_urls(request.url, max_pages=request.max_pages)
    tried_urls: set[str] = set()

    for index, candidate_url in enumerate(candidate_urls, start=1):
        tried_urls.add(candidate_url)
        try:
            _emit_progress(
                progress_callback,
                stage="fetching",
                message=f"Fetching page {index}/{len(candidate_urls)}",
                progress=5 + min(index * 8, 20),
                current=index,
                total=len(candidate_urls),
            )
            html = scraper.fetch_html(candidate_url)
            _emit_progress(progress_callback, stage="extracting", message="Extracting reviews", progress=20)
            candidate_product_name = extract_product_name(html)
            product_name = _prefer_product_name(product_name, candidate_product_name)
            review_dicts = extract_reviews(html, candidate_url)

            if not review_dicts and candidate_url == request.url:
                review_links = discover_review_links(html, candidate_url)
                for link_index, review_link in enumerate(review_links, start=1):
                    if review_link in tried_urls:
                        continue
                    tried_urls.add(review_link)
                    _emit_progress(
                        progress_callback,
                        stage="fetching",
                        message=f"Following review page {link_index}/{len(review_links)}",
                        progress=20 + min(link_index * 4, 12),
                        current=link_index,
                        total=len(review_links),
                    )
                    try:
                        candidate_html = scraper.fetch_html(review_link)
                    except ScrapeError as exc:
                        fetch_errors.append(str(exc))
                        continue
                    candidate_product_name = _prefer_product_name(product_name, extract_product_name(candidate_html))
                    candidate_reviews = extract_reviews(candidate_html, review_link)
                    if candidate_reviews:
                        product_name = _prefer_product_name(product_name, candidate_product_name)
                        review_dicts = candidate_reviews
                        break

            if review_dicts:
                for item in review_dicts:
                    record = _dict_to_record(item, product_name)
                    key = record.dedupe_key()
                    if key in seen_review_keys:
                        continue
                    seen_review_keys.add(key)
                    review_records.append(record)
        except ScrapeError as exc:
            fetch_errors.append(str(exc))
            continue

    if not review_records and request.url not in tried_urls:
        try:
            html = scraper.fetch_html(request.url)
            candidate_product_name = extract_product_name(html)
            product_name = _prefer_product_name(product_name, candidate_product_name)
            review_dicts = extract_reviews(html, request.url)
            for item in review_dicts:
                record = _dict_to_record(item, product_name)
                key = record.dedupe_key()
                if key in seen_review_keys:
                    continue
                seen_review_keys.add(key)
                review_records.append(record)
        except ScrapeError as exc:
            fetch_errors.append(str(exc))

    if review_records:
        _emit_progress(
            progress_callback,
            stage="extracting",
            message=f"Extracted {len(review_records)} review{'s' if len(review_records) != 1 else ''}",
            progress=35,
            current=len(review_records),
            total=len(review_records),
        )
    elif fetch_errors:
        cached = _get_cached_scrape(request.url)
        if cached is None:
            raise ScrapeError(fetch_errors[0])
        product_name, review_records = cached
        _emit_progress(
            progress_callback,
            stage="extracting",
            message="Using reviews from a recent successful scrape",
            progress=35,
            current=len(review_records),
            total=len(review_records),
        )
    else:
        cached = _get_cached_scrape(request.url)
        if cached is not None:
            product_name, review_records = cached
            _emit_progress(
                progress_callback,
                stage="extracting",
                message="No fresh reviews found, using a recent successful scrape",
                progress=35,
                current=len(review_records),
                total=len(review_records),
            )
        else:
            _emit_progress(
                progress_callback,
                stage="extracting",
                message="Extracted 0 reviews",
                progress=35,
                current=0,
                total=0,
            )

    total_reviews = max(len(review_records), 1)
    for index, review in enumerate(review_records, start=1):
        _emit_progress(
            progress_callback,
            stage="preprocessing",
            message=f"Preprocessing review {index}/{len(review_records)}",
            progress=35 + int((index / total_reviews) * 15),
            current=index,
            total=len(review_records),
        )
        prepare_review(review, processing_config)

    if review_records:
        _set_cached_scrape(request.url, product_name, review_records)

    overall_summary = None
    overall_sentiment = None
    overall_key_points = None
    overall_error = None

    if not request.skip_llm and review_records:
        api_key = os.getenv(request.api_key_env)
        if not api_key:
            raise PipelineValidationError(f"No API key found in environment variable {request.api_key_env}.")
        summarizer = ReviewSummarizer(
            LLMConfig(
                model=request.model,
                api_key=api_key,
                base_url=request.base_url,
                temperature=request.temperature,
                top_p=request.top_p,
                max_output_tokens=request.max_output_tokens,
                requests_per_minute=request.llm_rpm,
                max_retries=request.llm_retries,
                extra_body=parse_extra_body(request.extra_body_json),
            )
        )
        _emit_progress(
            progress_callback,
            stage="summarizing",
            message="Summarizing product reviews",
            progress=55,
            current=1,
            total=1,
        )
        try:
            overall_result = summarizer.summarize_product(product_name, review_records)
            overall_summary = overall_result.summary
            overall_sentiment = overall_result.sentiment
            overall_key_points = overall_result.key_points
        except Exception as exc:  # pragma: no cover
            overall_error = str(exc)
        _emit_progress(
            progress_callback,
            stage="summarizing",
            message="Finished product summary",
            progress=95,
            current=1,
            total=1,
        )

    _emit_progress(
        progress_callback,
        stage="completed",
        message=f"Completed {len(review_records)} review{'s' if len(review_records) != 1 else ''}",
        progress=100,
        current=len(review_records),
        total=len(review_records),
    )

    return PipelineResult(
        product_name=product_name,
        reviews=review_records,
        overall_summary=overall_summary,
        overall_sentiment=overall_sentiment,
        overall_key_points=overall_key_points,
        overall_error=overall_error,
    )


def prepare_review(review: ReviewRecord, config: ProcessingConfig) -> None:
    review.cleaned_review_text = clean_review_text(review.review_text)
    review.review_chunks = chunk_text(
        review.cleaned_review_text,
        model=config.tokenizer_model,
        max_tokens=config.max_input_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )


def parse_extra_body(extra_body_json: str | None) -> dict[str, Any] | None:
    if not extra_body_json:
        return None
    try:
        return json.loads(extra_body_json)
    except json.JSONDecodeError as exc:
        raise PipelineValidationError(f"Invalid extra_body_json payload: {exc}") from exc


def _dict_to_record(review: dict[str, Any], product_name: str | None) -> ReviewRecord:
    return ReviewRecord(
        source_url=str(review.get("source_url") or ""),
        page_url=str(review.get("page_url") or review.get("source_url") or ""),
        product_name=review.get("product_name") or product_name,
        author=review.get("author") or "Anonymous",
        rating=review.get("rating"),
        review_date=review.get("review_date"),
        title=review.get("title"),
        review_text=str(review.get("review_text") or "").strip(),
        raw_metadata=review.get("raw_metadata") or {},
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_model_choice(model: str | None, base_url: str | None, api_key_env: str | None) -> str:
    text = (model or "").strip()
    if not text:
        return text

    is_nvidia_profile = (base_url or "").strip() == NVIDIA_BASE_URL or (api_key_env or "").strip() == "NVIDIA_API_KEY"
    if is_nvidia_profile and text in NVIDIA_DEPRECATED_MODEL_ALIASES:
        return NVIDIA_DEFAULT_MODEL
    return text


def _prefer_product_name(current: str | None, candidate: str | None) -> str | None:
    candidate_text = (candidate or "").strip()
    current_text = (current or "").strip()
    if not candidate_text:
        return current
    if not current_text:
        return candidate_text
    if _is_generic_product_name(current_text) and not _is_generic_product_name(candidate_text):
        return candidate_text
    return current_text


def _is_generic_product_name(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {
        "",
        "amazon sign-in",
        "sign in",
        "best buy",
        "flipkart",
    }


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    progress: int,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if callback is None:
        return
    payload: dict[str, Any] = {
        "stage": stage,
        "message": message,
        "progress": max(0, min(progress, 100)),
    }
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    callback(payload)


def _get_cached_scrape(url: str) -> tuple[str | None, list[ReviewRecord]] | None:
    now = time.time()
    with _SCRAPE_CACHE_LOCK:
        entry = _SCRAPE_CACHE.get(url)
        if entry is None:
            return None
        if now - entry["timestamp"] > SCRAPE_CACHE_TTL_SECONDS:
            _SCRAPE_CACHE.pop(url, None)
            return None
        return entry["product_name"], copy.deepcopy(entry["reviews"])


def _set_cached_scrape(url: str, product_name: str | None, reviews: list[ReviewRecord]) -> None:
    with _SCRAPE_CACHE_LOCK:
        _SCRAPE_CACHE[url] = {
            "timestamp": time.time(),
            "product_name": product_name,
            "reviews": copy.deepcopy(reviews),
        }
