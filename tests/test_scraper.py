from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from review_scraper.models import ScraperConfig
from review_scraper.scraper import ReviewScraper, ScrapeError, build_candidate_urls, normalize_product_url


class ScraperTests(unittest.TestCase):
    def test_normalize_bestbuy_product_url_to_site_url(self):
        normalized = normalize_product_url(
            "https://www.bestbuy.com/product/shokz-openfit-pro-open-ear-true-wireless-bluetooth-active-noise-reduction-earbuds-black/J3GWRW4HCC/sku/6665563"
        )

        self.assertEqual(
            normalized,
            "https://www.bestbuy.com/site/shokz-openfit-pro-open-ear-true-wireless-bluetooth-active-noise-reduction-earbuds-black/6665563.p?skuId=6665563",
        )

    def test_normalize_bestbuy_product_id_url_to_site_url(self):
        normalized = normalize_product_url(
            "https://www.bestbuy.com/product/lg-65-class-c5-series-oled-evo-ai-4k-uhd-smart-webos-tv-2025/JJ8VPZTRG6"
        )

        self.assertEqual(
            normalized,
            "https://www.bestbuy.com/site/lg-65-class-c5-series-oled-evo-ai-4k-uhd-smart-webos-tv-2025/JJ8VPZTRG6.p",
        )

    def test_build_bestbuy_review_page_candidates(self):
        candidates = build_candidate_urls(
            "https://www.bestbuy.com/site/shokz-openfit-pro-open-ear-true-wireless-bluetooth-active-noise-reduction-earbuds-black/6665563.p?skuId=6665563",
            max_pages=2,
        )

        self.assertEqual(
            candidates,
            [
                "https://www.bestbuy.com/site/shokz-openfit-pro-open-ear-true-wireless-bluetooth-active-noise-reduction-earbuds-black/6665563.p?skuId=6665563",
                "https://www.bestbuy.com/site/reviews/shokz-openfit-pro-open-ear-true-wireless-bluetooth-active-noise-reduction-earbuds-black/6665563",
                "https://www.bestbuy.com/site/reviews/shokz-openfit-pro-open-ear-true-wireless-bluetooth-active-noise-reduction-earbuds-black/6665563?page=2",
            ],
        )

    def test_build_amazon_review_page_candidates(self):
        candidates = build_candidate_urls(
            "https://www.amazon.in/Apple-2026-MacBook-Laptop-chip/dp/B0GR68779Y/",
            max_pages=2,
        )

        self.assertEqual(
            candidates,
            [
                "https://www.amazon.in/Apple-2026-MacBook-Laptop-chip/dp/B0GR68779Y/",
                "https://www.amazon.in/product-reviews/B0GR68779Y/?pageNumber=1",
                "https://www.amazon.in/product-reviews/B0GR68779Y/?pageNumber=2",
            ],
        )

    def test_fetch_html_returns_text_for_http_200(self):
        scraper = ReviewScraper(ScraperConfig())
        response = Mock(status_code=200, text="<html>ok</html>")

        with patch.object(scraper.session, "get", return_value=response):
            html = scraper.fetch_html("https://example.com/product")

        self.assertEqual(html, "<html>ok</html>")

    def test_fetch_html_raises_clear_error_for_non_200(self):
        scraper = ReviewScraper(ScraperConfig(enable_browser_fallback=False))
        response = Mock(status_code=403, text="forbidden")

        with patch.object(scraper.session, "get", return_value=response):
            with self.assertRaises(ScrapeError) as ctx:
                scraper.fetch_html("https://example.com/product")

        self.assertIn("HTTP 403", str(ctx.exception))

    def test_fetch_html_uses_configured_timeout(self):
        scraper = ReviewScraper(ScraperConfig(request_timeout=45))
        response = Mock(status_code=200, text="<html>ok</html>")

        with patch.object(scraper.session, "get", return_value=response) as get_mock:
            scraper.fetch_html("https://example.com/product")

        self.assertEqual(get_mock.call_args.kwargs["timeout"], 45)

    def test_fetch_html_uses_browser_fallback_on_request_exception(self):
        scraper = ReviewScraper(ScraperConfig(enable_browser_fallback=True))

        with patch.object(
            scraper.session,
            "get",
            side_effect=requests.ReadTimeout("timed out"),
        ), patch.object(
            scraper._browser_fetcher,
            "fetch_html",
            return_value="<html>browser ok</html>",
        ) as browser_fetch:
            html = scraper.fetch_html("https://example.com/product")

        self.assertEqual(html, "<html>browser ok</html>")
        browser_fetch.assert_called_once()

    def test_fetch_html_uses_browser_fallback_on_blocked_status(self):
        scraper = ReviewScraper(ScraperConfig(enable_browser_fallback=True))
        response = Mock(status_code=403, text="blocked")

        with patch.object(scraper.session, "get", return_value=response), patch.object(
            scraper._browser_fetcher,
            "fetch_html",
            return_value="<html>browser ok</html>",
        ) as browser_fetch:
            html = scraper.fetch_html("https://example.com/product")

        self.assertEqual(html, "<html>browser ok</html>")
        browser_fetch.assert_called_once()

    def test_fetch_html_uses_browser_fallback_on_http_529(self):
        scraper = ReviewScraper(ScraperConfig(enable_browser_fallback=True))
        response = Mock(status_code=529, text="site overloaded")

        with patch.object(scraper.session, "get", return_value=response), patch.object(
            scraper._browser_fetcher,
            "fetch_html",
            return_value="<html>browser ok</html>",
        ) as browser_fetch:
            html = scraper.fetch_html("https://example.com/product")

        self.assertEqual(html, "<html>browser ok</html>")
        browser_fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
