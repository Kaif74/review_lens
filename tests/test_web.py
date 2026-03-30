from __future__ import annotations

import unittest
from unittest.mock import patch

import review_scraper.service as service_module
from review_scraper.models import ReviewRecord
from review_scraper.service import PipelineResult, get_runtime_defaults, parse_pipeline_request, run_review_pipeline
from review_scraper.scraper import ScrapeError
from review_scraper.web import JOBS, JOBS_LOCK, create_app


class ImmediateThread:
    def __init__(self, target=None, args=None, kwargs=None, daemon=None):
        self.target = target
        self.args = args or ()
        self.kwargs = kwargs or {}

    def start(self):
        if self.target:
            self.target(*self.args, **self.kwargs)


class WebAppTests(unittest.TestCase):
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "secret"}, clear=True)
    def test_runtime_defaults_prefer_nvidia_profile(self):
        defaults = get_runtime_defaults()

        self.assertEqual(defaults["api_key_env"], "NVIDIA_API_KEY")
        self.assertEqual(defaults["model"], "deepseek-ai/deepseek-v3.1")
        self.assertEqual(defaults["base_url"], "https://integrate.api.nvidia.com/v1")

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "secret"}, clear=True)
    def test_parse_pipeline_request_remaps_retired_nvidia_model(self):
        request = parse_pipeline_request(
            {
                "url": "https://example.com/product",
                "model": "deepseek-ai/deepseek-r1",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key_env": "NVIDIA_API_KEY",
            }
        )

        self.assertEqual(request.model, "deepseek-ai/deepseek-v3.1")

    def setUp(self):
        with JOBS_LOCK:
            JOBS.clear()
        with service_module._SCRAPE_CACHE_LOCK:
            service_module._SCRAPE_CACHE.clear()
        self.app = create_app()
        self.client = self.app.test_client()

    def test_index_page_renders(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Review Lens", response.data)

    @patch("review_scraper.web.run_review_pipeline")
    def test_reviews_endpoint_returns_json(self, run_review_pipeline_mock):
        run_review_pipeline_mock.return_value = PipelineResult(
            product_name="Acme Headphones",
            reviews=[
                ReviewRecord(
                    source_url="https://example.com/product",
                    page_url="https://example.com/reviews",
                    product_name="Acme Headphones",
                    author="Taylor",
                    rating=5.0,
                    review_text="Excellent battery life and clear sound.",
                    llm_summary="Strong audio and battery performance.",
                    llm_sentiment="positive",
                    llm_key_points=["Long battery life", "Clear sound"],
                )
            ],
            overall_summary="Customers mostly praise sound quality and battery life.",
            overall_sentiment="Positive",
            overall_key_points=["Strong sound", "Long battery life", "Comfortable fit"],
        )

        response = self.client.post("/api/reviews", json={"url": "https://example.com/product", "skip_llm": True})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["product_name"], "Acme Headphones")
        self.assertEqual(payload["review_count"], 1)
        self.assertEqual(payload["reviews"][0]["author"], "Taylor")
        self.assertEqual(payload["overall_sentiment"], "Positive")

    def test_reviews_endpoint_validates_url(self):
        response = self.client.post("/api/reviews", json={"url": ""})

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("URL", payload["error"].upper())

    @patch("review_scraper.web.run_review_pipeline")
    def test_reviews_endpoint_returns_422_when_no_reviews_found(self, run_review_pipeline_mock):
        run_review_pipeline_mock.return_value = PipelineResult(product_name="Acme", reviews=[])

        response = self.client.post("/api/reviews", json={"url": "https://example.com/product"})

        self.assertEqual(response.status_code, 422)
        payload = response.get_json()
        self.assertIn("No reviews found", payload["error"])

    @patch("review_scraper.web.threading.Thread", ImmediateThread)
    @patch("review_scraper.web.run_review_pipeline")
    def test_async_job_progress_endpoint_returns_completed_result(self, run_review_pipeline_mock):
        run_review_pipeline_mock.return_value = PipelineResult(
            product_name="Acme Headphones",
            reviews=[
                ReviewRecord(
                    source_url="https://example.com/product",
                    page_url="https://example.com/product",
                    product_name="Acme Headphones",
                    author="Taylor",
                    review_text="Excellent battery life and clear sound.",
                )
            ],
        )

        start_response = self.client.post("/api/reviews/start", json={"url": "https://example.com/product", "skip_llm": True})
        self.assertEqual(start_response.status_code, 202)
        job_id = start_response.get_json()["job_id"]

        progress_response = self.client.get(f"/api/reviews/progress/{job_id}")
        self.assertEqual(progress_response.status_code, 200)
        payload = progress_response.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["review_count"], 1)

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "secret"}, clear=True)
    @patch("review_scraper.service.ReviewSummarizer")
    @patch("review_scraper.service.extract_reviews")
    @patch("review_scraper.service.extract_product_name")
    @patch("review_scraper.service.ReviewScraper.fetch_html")
    def test_ai_run_can_reuse_recent_successful_scrape(
        self,
        fetch_html_mock,
        extract_product_name_mock,
        extract_reviews_mock,
        summarizer_cls_mock,
    ):
        extract_product_name_mock.return_value = "Acme Headphones"
        extract_reviews_mock.return_value = [
            {
                "source_url": "https://example.com/product",
                "page_url": "https://example.com/product",
                "author": "Taylor",
                "rating": 5.0,
                "review_text": "Excellent battery life and clear sound.",
                "title": "Great sound",
            }
        ]

        first_request = parse_pipeline_request({"url": "https://example.com/product", "skip_llm": True})
        first_result = run_review_pipeline(first_request)
        self.assertEqual(len(first_result.reviews), 1)

        fetch_html_mock.side_effect = ScrapeError("Fetch failed")
        summarizer = summarizer_cls_mock.return_value
        summarizer.summarize_product.return_value.summary = "Positive overall."
        summarizer.summarize_product.return_value.sentiment = "Positive"
        summarizer.summarize_product.return_value.key_points = ["Battery life", "Clear sound", "Worth buying"]

        second_request = parse_pipeline_request({"url": "https://example.com/product", "skip_llm": False})
        second_result = run_review_pipeline(second_request)

        self.assertEqual(len(second_result.reviews), 1)
        self.assertEqual(second_result.overall_sentiment, "Positive")

    @patch("review_scraper.service.discover_review_links")
    @patch("review_scraper.service.extract_reviews")
    @patch("review_scraper.service.extract_product_name")
    @patch("review_scraper.service.ReviewScraper.fetch_html")
    def test_pipeline_follows_review_link_when_product_page_has_no_reviews(
        self,
        fetch_html_mock,
        extract_product_name_mock,
        extract_reviews_mock,
        discover_review_links_mock,
    ):
        fetch_html_mock.side_effect = ["<html>product</html>", "<html>reviews</html>"]
        extract_product_name_mock.side_effect = ["Acme Headphones", "Acme Headphones"]
        extract_reviews_mock.side_effect = [
            [],
            [
                {
                    "source_url": "https://example.com/product",
                    "page_url": "https://example.com/reviews",
                    "author": "Taylor",
                    "rating": 5.0,
                    "review_text": "Excellent battery life and clear sound.",
                }
            ],
        ]
        discover_review_links_mock.return_value = ["https://example.com/reviews"]

        request = parse_pipeline_request({"url": "https://example.com/product", "skip_llm": True})
        result = run_review_pipeline(request)

        self.assertEqual(len(result.reviews), 1)
        self.assertEqual(result.reviews[0].author, "Taylor")


if __name__ == "__main__":
    unittest.main()
