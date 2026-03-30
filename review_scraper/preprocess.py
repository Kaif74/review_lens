from __future__ import annotations

import html
import re
import unicodedata

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TRAILING_UI_ARTIFACT_RE = re.compile(r"(?:\s*(?:read more|show more|read less|show less)\s*)+$", re.IGNORECASE)


def clean_review_text(text: str) -> str:
    if not text:
        return ""
    cleaned = html.unescape(text)
    cleaned = cleaned.replace("\x00", " ")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.replace("\u200b", " ").replace("\ufeff", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = TRAILING_UI_ARTIFACT_RE.sub("", cleaned)
    return cleaned.strip()


def count_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return max(1, len(text) // 4)
    return len(encoding.encode(text))


def chunk_text(text: str, model: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    cleaned = clean_review_text(text)
    if not cleaned:
        return []
    if count_tokens(cleaned, model) <= max_tokens:
        return [cleaned]

    sentences = [part.strip() for part in SENTENCE_RE.split(cleaned) if part.strip()]
    if not sentences:
        return [cleaned]

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and count_tokens(candidate, model) > max_tokens:
            chunks.append(" ".join(current).strip())
            current = _overlap_tail(current, model, max_tokens=overlap_tokens)
            current.append(sentence)
        else:
            current.append(sentence)

        if count_tokens(" ".join(current), model) > max_tokens:
            words = sentence.split()
            current = []
            bucket: list[str] = []
            for word in words:
                candidate = " ".join(bucket + [word]).strip()
                if bucket and count_tokens(candidate, model) > max_tokens:
                    chunks.append(" ".join(bucket).strip())
                    bucket = [word]
                else:
                    bucket.append(word)
            if bucket:
                chunks.append(" ".join(bucket).strip())

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _overlap_tail(sentences: list[str], model: str, max_tokens: int) -> list[str]:
    if max_tokens <= 0:
        return []
    tail: list[str] = []
    for sentence in reversed(sentences):
        candidate = " ".join([sentence] + tail).strip()
        if tail and count_tokens(candidate, model) > max_tokens:
            break
        tail.insert(0, sentence)
    return tail
