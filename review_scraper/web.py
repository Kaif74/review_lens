from __future__ import annotations

import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from .exporters import reviews_to_csv_text
from .scraper import ScrapeError
from .service import PipelineValidationError, get_runtime_defaults, parse_pipeline_request, run_review_pipeline


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__, template_folder="templates")

    @app.get("/")
    def index():
        return render_template("index.html", defaults=get_runtime_defaults())

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.post("/api/reviews")
    def reviews():
        payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
        payload = payload or {}

        try:
            pipeline_request = parse_pipeline_request(payload)
            result = run_review_pipeline(pipeline_request)
        except PipelineValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except ScrapeError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"Unexpected server error: {exc}"}), 500

        if not result.reviews:
            return jsonify({"error": "No reviews found on this page. Try a direct reviews URL."}), 422

        wants_csv = str(payload.get("response_format", "")).lower() == "csv"
        if wants_csv:
            return app.response_class(
                reviews_to_csv_text(result.reviews),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=reviews.csv"},
            )

        return jsonify(
            {
                "product_name": result.product_name,
                "review_count": len(result.reviews),
                "overall_summary": result.overall_summary,
                "overall_sentiment": result.overall_sentiment,
                "overall_key_points": result.overall_key_points or [],
                "overall_error": result.overall_error,
                "reviews": [review.to_flat_dict() for review in result.reviews],
            }
        )

    @app.post("/api/reviews/start")
    def start_reviews_job():
        payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
        payload = payload or {}

        try:
            pipeline_request = parse_pipeline_request(payload)
        except PipelineValidationError as exc:
            return jsonify({"error": str(exc)}), 400

        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "queued",
                "message": "Queued",
                "progress": 0,
                "result": None,
                "error": None,
            }

        thread = threading.Thread(
            target=_run_reviews_job,
            args=(job_id, pipeline_request),
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.get("/api/reviews/progress/<job_id>")
    def reviews_progress(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                return jsonify({"error": "Job not found."}), 404
            payload = dict(job)
        return jsonify(payload)

    return app


app = create_app()


def main() -> int:
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
    return 0


def _run_reviews_job(job_id, pipeline_request) -> None:
    _update_job(job_id, status="running", message="Starting analysis", progress=1)

    def on_progress(update: dict) -> None:
        _update_job(
            job_id,
            status="running",
            message=update.get("message", "Working"),
            progress=update.get("progress", 0),
            stage=update.get("stage"),
            current=update.get("current"),
            total=update.get("total"),
        )

    try:
        result = run_review_pipeline(pipeline_request, progress_callback=on_progress)
        if not result.reviews:
            _update_job(
                job_id,
                status="failed",
                message="No reviews found on this page. Try a direct reviews URL.",
                progress=100,
                error="No reviews found on this page. Try a direct reviews URL.",
                code=422,
            )
            return

        _update_job(
            job_id,
            status="completed",
            message=f"Completed {len(result.reviews)} review{'s' if len(result.reviews) != 1 else ''}",
            progress=100,
            result={
                "product_name": result.product_name,
                "review_count": len(result.reviews),
                "overall_summary": result.overall_summary,
                "overall_sentiment": result.overall_sentiment,
                "overall_key_points": result.overall_key_points or [],
                "overall_error": result.overall_error,
                "reviews": [review.to_flat_dict() for review in result.reviews],
            },
            error=None,
        )
    except PipelineValidationError as exc:
        _update_job(job_id, status="failed", message=str(exc), progress=100, error=str(exc), code=400)
    except ScrapeError as exc:
        _update_job(job_id, status="failed", message=str(exc), progress=100, error=str(exc), code=502)
    except Exception as exc:  # pragma: no cover
        _update_job(
            job_id,
            status="failed",
            message=f"Unexpected server error: {exc}",
            progress=100,
            error=f"Unexpected server error: {exc}",
            code=500,
        )


def _update_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(
            job_id,
            {"status": "queued", "message": "Queued", "progress": 0, "result": None, "error": None},
        )
        job.update({key: value for key, value in updates.items() if value is not None})
