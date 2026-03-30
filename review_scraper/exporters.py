from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from .models import ReviewRecord


def reviews_to_rows(reviews: list[ReviewRecord]) -> list[dict]:
    return [review.to_flat_dict() for review in reviews]


def reviews_to_json_text(reviews: list[ReviewRecord]) -> str:
    return json.dumps(reviews_to_rows(reviews), indent=2, ensure_ascii=False)


def reviews_to_csv_text(reviews: list[ReviewRecord]) -> str:
    rows = reviews_to_rows(reviews)
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row.keys()})
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def export_reviews(reviews: list[ReviewRecord], output_path: str, fmt: str = "auto") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    chosen_format = fmt
    if fmt == "auto":
        chosen_format = output.suffix.lower().lstrip(".") or "json"
    if chosen_format not in {"json", "csv"}:
        raise ValueError("Output format must be one of: auto, json, csv")

    rows = reviews_to_rows(reviews)
    if chosen_format == "json":
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        output.write_text(reviews_to_csv_text(reviews), encoding="utf-8", newline="")
    return output
