from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReviewRecord:
    source_url: str
    page_url: str
    product_name: str | None = None
    review_id: str | None = None
    author: str | None = None
    rating: float | None = None
    review_date: str | None = None
    title: str | None = None
    review_text: str = ""
    verified_purchase: bool | None = None
    helpful_count: int | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    cleaned_review_text: str | None = None
    review_chunks: list[str] = field(default_factory=list)
    llm_summary: str | None = None
    llm_sentiment: str | None = None
    llm_key_points: list[str] = field(default_factory=list)
    llm_model: str | None = None
    llm_error: str | None = None

    def dedupe_key(self) -> tuple[str | None, str | None, str]:
        text_stub = (self.review_text or "").strip().lower()[:120]
        return (self.review_id, self.author, text_stub)

    def to_flat_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["review_chunks"] = "\n---CHUNK---\n".join(self.review_chunks)
        data["llm_key_points"] = " | ".join(self.llm_key_points)
        return data


@dataclass
class SummarizationResult:
    summary: str
    sentiment: str
    key_points: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 300
    requests_per_minute: int = 20
    max_retries: int = 4
    extra_body: dict[str, Any] | None = None


@dataclass
class ScraperConfig:
    request_timeout: int = 20
    min_delay_seconds: float = 0.8
    max_delay_seconds: float = 2.2
    max_retries: int = 4
    max_pages: int = 10
    enable_browser_fallback: bool = True
    browser_timeout_ms: int = 45000
    browser_headless: bool = True
    user_agents: list[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        ]
    )


@dataclass
class ProcessingConfig:
    tokenizer_model: str = "gpt-4o-mini"
    max_input_tokens: int = 1200
    chunk_overlap_tokens: int = 80
