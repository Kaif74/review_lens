from __future__ import annotations

import unittest

from review_scraper.adapters import discover_review_links, extract_product_name, extract_reviews
from review_scraper.preprocess import chunk_text, clean_review_text


JSON_LD_HTML = """
<html>
  <head>
    <meta property="og:title" content="Acme Headphones" />
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Acme Headphones",
        "review": [
          {
            "@type": "Review",
            "author": {"@type": "Person", "name": "Taylor"},
            "datePublished": "2026-01-05",
            "reviewBody": "Battery life is strong and audio is clear.",
            "reviewRating": {"@type": "Rating", "ratingValue": "5"}
          }
        ]
      }
    </script>
  </head>
</html>
"""


HTML_REVIEW_BLOCK = """
<html>
  <body>
    <h1>Simple Product</h1>
    <article>
      <div>4.0 out of 5 stars</div>
      <div class="author">Jordan</div>
      <time datetime="2026-02-14">2026-02-14</time>
      <p>This is a solid product for everyday use and the build quality feels reliable.</p>
    </article>
  </body>
</html>
"""

FLIPKART_REVIEW_HTML = """
<html>
  <body>
    <div>
      <div>5★</div>
      <div class="_6K-7Co">Highly recommended</div>
      <p>Very sturdy cycle and good value for money. Read more</p>
      <div>Rahul Sharma Verified Buyer 2 months ago</div>
    </div>
  </body>
</html>
"""

FLIPKART_BLOCK_HTML = """
<html>
  <body>
    <div class="_16PBlm">
      <div class="_3LWZlK _1BLPMq">5</div>
      <p class="_2-N8zT">Brilliant</p>
      <div class="t-ZTKy">
        <div>Excellent camera and battery backup.</div>
      </div>
      <p class="_2sc7ZR">Aman Verma</p>
      <p>Verified Buyer</p>
      <p>3 months ago</p>
    </div>
  </body>
</html>
"""

PRODUCT_PAGE_WITH_REVIEW_LINK = """
<html>
  <body>
    <h1>Simple Product</h1>
    <a href="/reviews/simple-product">Show all reviews</a>
  </body>
</html>
"""


class ReviewPipelineTests(unittest.TestCase):
    def test_extract_product_name_prefers_meta_or_json_ld(self):
        product_name = extract_product_name(JSON_LD_HTML)
        self.assertEqual(product_name, "Acme Headphones")

    def test_extract_reviews_reads_json_ld_reviews(self):
        reviews = extract_reviews(JSON_LD_HTML, "https://example.com/product")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["author"], "Taylor")
        self.assertEqual(reviews[0]["rating"], 5.0)
        self.assertIn("Battery life is strong", reviews[0]["review_text"])

    def test_extract_reviews_falls_back_to_html_heuristics(self):
        reviews = extract_reviews(HTML_REVIEW_BLOCK, "https://example.com/product")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["author"], "Jordan")
        self.assertEqual(reviews[0]["rating"], 4.0)
        self.assertEqual(reviews[0]["review_date"], "2026-02-14")

    def test_extract_reviews_reads_flipkart_style_review_blocks(self):
        reviews = extract_reviews(
            FLIPKART_REVIEW_HTML,
            "https://www.flipkart.com/leader-scout/product-reviews/itmbbb147868db88?pid=CCEFSESZ88VXH4FD&page=1",
        )
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["author"], "Rahul Sharma")
        self.assertEqual(reviews[0]["rating"], 5.0)
        self.assertEqual(reviews[0]["title"], "Highly recommended")
        self.assertIn("Very sturdy cycle", reviews[0]["review_text"])

    def test_extract_reviews_reads_flipkart_classic_review_blocks(self):
        reviews = extract_reviews(
            FLIPKART_BLOCK_HTML,
            "https://www.flipkart.com/apple-iphone-17-pro-cosmic-orange-256-gb/product-reviews/itm76fe37ca9ea8c?pid=MOBHFN6YR8HF5BQ9&page=1",
        )
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["author"], "Aman Verma")
        self.assertEqual(reviews[0]["rating"], 5.0)
        self.assertEqual(reviews[0]["title"], "Brilliant")
        self.assertIn("Excellent camera", reviews[0]["review_text"])

    def test_discover_review_links_finds_review_page_candidates(self):
        links = discover_review_links(PRODUCT_PAGE_WITH_REVIEW_LINK, "https://example.com/product")
        self.assertEqual(links, ["https://example.com/reviews/simple-product"])

    def test_preprocess_cleans_and_chunks_long_text(self):
        raw_text = (
            "Excellent sound quality!   "
            "Battery life lasted all day.\n\n"
            "The companion app is simple to use and pairing was fast. "
            "I would buy it again. " * 20
        )

        cleaned = clean_review_text(raw_text)
        chunks = chunk_text(cleaned, model="gpt-4o-mini", max_tokens=60, overlap_tokens=10)

        self.assertNotIn("  ", cleaned)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_clean_review_text_strips_trailing_read_more_artifacts(self):
        cleaned = clean_review_text("Excellent phone overall. Read more")
        self.assertEqual(cleaned, "Excellent phone overall.")


if __name__ == "__main__":
    unittest.main()
