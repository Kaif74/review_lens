from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .models import LLMConfig, ReviewRecord, SummarizationResult
from .preprocess import chunk_text


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
SENTIMENT_RE = re.compile(r"\b(Positive|Negative|Mixed)\b", re.IGNORECASE)


class ReviewSummarizer:
    def __init__(self, config: LLMConfig):
        self.config = config
        client_kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = OpenAI(**client_kwargs)
        self._min_interval = 60.0 / max(config.requests_per_minute, 1)
        self._last_request_ts = 0.0

    def summarize_review(self, review: ReviewRecord) -> SummarizationResult:
        text = review.cleaned_review_text or review.review_text
        chunks = review.review_chunks or [text]

        chunk_results = [
            self._summarize_text(self._build_prompt(review, chunk, index, len(chunks)))
            for index, chunk in enumerate(chunks, start=1)
        ]
        if len(chunk_results) == 1:
            return chunk_results[0]
        return self._summarize_text(self._build_synthesis_prompt(review, chunk_results))

    def summarize_product(self, product_name: str | None, reviews: list[ReviewRecord]) -> SummarizationResult:
        combined_reviews = []
        for index, review in enumerate(reviews, start=1):
            body = (review.cleaned_review_text or review.review_text or "").strip()
            if not body:
                continue
            title = (review.title or "").strip()
            rating = f"{review.rating}/5" if review.rating is not None else "unknown"
            author = (review.author or "Anonymous").strip()
            prefix = f"Review {index} | rating={rating} | author={author}"
            if title:
                prefix += f" | title={title}"
            combined_reviews.append(f"{prefix}\n{body}")

        corpus = "\n\n".join(combined_reviews).strip()
        if not corpus:
            return SummarizationResult(
                summary="Summary unavailable.",
                sentiment="Mixed",
                key_points=["Point unavailable.", "Point unavailable.", "Point unavailable."],
            )

        chunks = chunk_text(corpus, model=self.config.model, max_tokens=2200, overlap_tokens=120)
        chunk_results = [
            self._summarize_text(self._build_collection_prompt(product_name, chunk, index, len(chunks)))
            for index, chunk in enumerate(chunks, start=1)
        ]
        if len(chunk_results) == 1:
            return chunk_results[0]
        return self._summarize_text(self._build_collection_synthesis_prompt(product_name, chunk_results, len(reviews)))

    def _build_prompt(self, review: ReviewRecord, chunk: str, chunk_index: int, total_chunks: int) -> str:
        metadata = {
            "product_name": review.product_name,
            "author": review.author,
            "rating": review.rating,
            "date": review.review_date,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }
        return (
            "Analyze this product review and return valid JSON.\n"
            "Required JSON keys: sentiment, summary, key_points.\n"
            'sentiment must be exactly one of: "Positive", "Negative", "Mixed".\n'
            "summary must be 2 to 3 sentences.\n"
            "key_points must be a JSON array with exactly 3 short bullet strings.\n\n"
            f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Review text:\n{chunk}"
        )

    def _build_synthesis_prompt(
        self,
        review: ReviewRecord,
        chunk_results: list[SummarizationResult],
    ) -> str:
        payload = [
            {"sentiment": item.sentiment, "summary": item.summary, "key_points": item.key_points}
            for item in chunk_results
        ]
        return (
            "Combine these chunk-level analyses of one product review and return valid JSON.\n"
            "Required JSON keys: sentiment, summary, key_points.\n"
            'sentiment must be exactly one of: "Positive", "Negative", "Mixed".\n'
            "summary must be 2 to 3 sentences.\n"
            "key_points must be a JSON array with exactly 3 short bullet strings.\n\n"
            f"Review title: {review.title or ''}\n"
            f"Chunk analyses: {json.dumps(payload, ensure_ascii=False)}"
        )

    def _build_collection_prompt(
        self,
        product_name: str | None,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        metadata = {
            "product_name": product_name,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }
        return (
            "Analyze these customer reviews for one product and return valid JSON.\n"
            "Required JSON keys: sentiment, summary, key_points.\n"
            'sentiment must be exactly one of: "Positive", "Negative", "Mixed".\n'
            "summary must be 2 to 4 sentences describing the overall customer consensus.\n"
            "key_points must be a JSON array with exactly 3 short bullet strings covering the biggest positives or negatives.\n\n"
            f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Customer reviews:\n{chunk}"
        )

    def _build_collection_synthesis_prompt(
        self,
        product_name: str | None,
        chunk_results: list[SummarizationResult],
        review_count: int,
    ) -> str:
        payload = [
            {"sentiment": item.sentiment, "summary": item.summary, "key_points": item.key_points}
            for item in chunk_results
        ]
        return (
            "Combine these chunk-level summaries of customer reviews for one product and return valid JSON.\n"
            "Required JSON keys: sentiment, summary, key_points.\n"
            'sentiment must be exactly one of: "Positive", "Negative", "Mixed".\n'
            "summary must be 2 to 4 sentences describing the overall customer consensus.\n"
            "key_points must be a JSON array with exactly 3 short bullet strings.\n\n"
            f"Product name: {product_name or ''}\n"
            f"Review count: {review_count}\n"
            f"Chunk summaries: {json.dumps(payload, ensure_ascii=False)}"
        )

    def _summarize_text(self, prompt: str) -> SummarizationResult:
        attempt = 0
        while True:
            attempt += 1
            self._respect_rate_limit()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You summarize customer reviews in compact JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_output_tokens,
                    extra_body=self.config.extra_body,
                )
                content = self._extract_message_content(response)
                return self._parse_result(content)
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                if attempt > self.config.max_retries:
                    raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
                time.sleep(min(2 ** attempt, 20))
            except APIStatusError as exc:
                if exc.status_code not in {408, 409, 429, 500, 502, 503, 504} or attempt > self.config.max_retries:
                    raise RuntimeError(f"LLM request failed with status {exc.status_code}: {exc}") from exc
                time.sleep(min(2 ** attempt, 20))

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _extract_message_content(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _parse_result(self, text: str) -> SummarizationResult:
        match = JSON_OBJECT_RE.search(text)
        if match:
            try:
                payload = json.loads(match.group(0))
                return self._payload_to_result(payload)
            except json.JSONDecodeError:
                pass
        return self._plain_text_to_result(text)

    def _payload_to_result(self, payload: dict[str, Any]) -> SummarizationResult:
        sentiment = str(payload.get("sentiment") or "Mixed").strip().title()
        if sentiment not in {"Positive", "Negative", "Mixed"}:
            sentiment = "Mixed"
        summary = str(payload.get("summary") or "").strip()
        key_points = payload.get("key_points") or []
        if isinstance(key_points, str):
            key_points = [part.strip("- ").strip() for part in key_points.splitlines() if part.strip()]
        if not isinstance(key_points, list):
            key_points = []
        points = [str(point).strip() for point in key_points if str(point).strip()][:3]
        if len(points) < 3:
            points.extend(["Point unavailable."] * (3 - len(points)))
        return SummarizationResult(
            summary=summary or "Summary unavailable.",
            sentiment=sentiment,
            key_points=points,
        )

    def _plain_text_to_result(self, text: str) -> SummarizationResult:
        cleaned = text.strip()
        sentiment_match = SENTIMENT_RE.search(cleaned)
        sentiment = sentiment_match.group(1).title() if sentiment_match else "Mixed"

        summary_match = re.search(r"summary\s*:?\s*(.+?)(?:key[_ ]?points|$)", cleaned, re.IGNORECASE | re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
        else:
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            summary = " ".join(sentence.strip() for sentence in sentences[:3] if sentence.strip())

        bullets = re.findall(r"^[\-\*\u2022]\s*(.+)$", cleaned, flags=re.MULTILINE)
        if not bullets:
            key_section = re.search(r"key[_ ]?points\s*:?\s*(.+)$", cleaned, re.IGNORECASE | re.DOTALL)
            if key_section:
                bullets = [part.strip() for part in re.split(r"[;\n]", key_section.group(1)) if part.strip()]
        points = bullets[:3] if bullets else ["Point unavailable."] * 3
        if len(points) < 3:
            points.extend(["Point unavailable."] * (3 - len(points)))

        return SummarizationResult(
            summary=summary or "Summary unavailable.",
            sentiment=sentiment,
            key_points=points,
        )
