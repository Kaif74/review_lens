# Review Lens

Review Lens is a Flask web app that scrapes reachable customer reviews from product pages, cleans the review text, and produces one overall AI summary for the product using an OpenAI-compatible API.

## Current Behavior

- Accepts a product URL in the browser UI.
- Tries retailer-specific review-page resolution for Amazon, Flipkart, and Best Buy before falling back to generic review-link discovery.
- Uses `requests` first and falls back to Playwright when a site times out or blocks the plain HTTP request.
- Extracts review text plus metadata such as title, rating, author, and date.
- Cleans text noise such as trailing `Read more` artifacts.
- Generates one product-level AI summary:
  - overall sentiment
  - overall summary
  - 3 key points
- Shows the raw reviews underneath as supporting evidence.
- Supports JSON and CSV download from the web UI.

## Important Limits

- Amazon may only expose the public reviews visible without login. If Amazon gates deeper review pages behind sign-in, this app cannot legally or reliably bypass that with an unauthenticated session.
- Big e-commerce sites can change markup or block automation at any time. The scraper is stronger than a generic HTML parser, but no local scraper can guarantee every request from every machine and IP will succeed.
- The app is optimized for a normal web server process. It is not a great fit for strict serverless limits when Playwright fallback is needed.

## Stack

- Backend: Flask
- Scraping: `requests`, `BeautifulSoup`, `lxml`
- Browser fallback: Playwright
- LLM: `openai` Python SDK against OpenAI-compatible APIs
- Text preprocessing: `tiktoken` with fallback token estimation

## Project Layout

```text
api/
  index.py
review_scraper/
  adapters.py
  browser.py
  exporters.py
  llm.py
  models.py
  preprocess.py
  scraper.py
  service.py
  web.py
  templates/
    index.html
tests/
  test_pipeline.py
  test_scraper.py
  test_web.py
.env.example
requirements.txt
vercel.json
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

2. Create your local environment file:

```bash
copy .env.example .env
```

3. Fill in your real values in `.env`.

Example NVIDIA-compatible setup:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
REVIEW_SCRAPER_MODEL=deepseek-ai/deepseek-v3.1
REVIEW_SCRAPER_BASE_URL=https://integrate.api.nvidia.com/v1
REVIEW_SCRAPER_API_KEY_ENV=NVIDIA_API_KEY
REVIEW_SCRAPER_TOKENIZER_MODEL=gpt-4o-mini
REVIEW_SCRAPER_REQUEST_TIMEOUT=30
```

## Run Locally

```bash
python -m review_scraper
```

Then open:

```text
http://127.0.0.1:8000
```

The app automatically loads variables from a root-level `.env` file.

## UI Flow

1. Paste a product URL.
2. Choose whether to skip AI summarization.
3. Optionally raise `Max review pages` or `HTTP timeout`.
4. Run analysis.
5. Review the overall product summary and the supporting raw reviews.

## Current Output

The UI and API return:

- `product_name`
- `review_count`
- `overall_summary`
- `overall_sentiment`
- `overall_key_points`
- `overall_error`
- `reviews`

Each item in `reviews` includes the extracted metadata and cleaned/raw review text.

## API Endpoints

- `GET /`
  - serves the browser UI
- `GET /api/health`
  - simple health check
- `POST /api/reviews`
  - synchronous scrape + summarize response
- `POST /api/reviews/start`
  - starts a background job
- `GET /api/reviews/progress/<job_id>`
  - polls job progress and returns the final result

## Supported Retailer Logic

### Amazon

- Tries the product page first.
- Can also generate direct `/product-reviews/<ASIN>` candidates.
- Best effort only for publicly reachable reviews.

### Flipkart

- Supports direct `product-reviews` URLs.
- Includes parsers for multiple Flipkart review page layouts, including older class-based review blocks and newer buyer-marker layouts.

### Best Buy

- Normalizes multiple Best Buy product URL shapes into canonical public product URLs.
- Generates direct Best Buy review page candidates.

## Hosting

Recommended:

- Render
- Railway

Why:

- this app can make long-running scraping requests
- Playwright fallback is much easier in a normal web service than in strict serverless runtimes

The repo still contains `api/index.py` and `vercel.json`, but the current implementation is better suited to container-style hosting than to Vercel serverless functions.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall review_scraper tests api
```

At the current implementation state, the automated test suite covers:

- JSON-LD extraction
- generic heuristic extraction
- Amazon review-page candidate generation
- Flipkart review parsing for multiple layouts
- Best Buy URL normalization and review-page candidate generation
- progress job behavior
- NVIDIA default configuration handling

## Example Test URLs

These are useful for local experimentation, but live retailer behavior can change:

- Best Buy:
  - `https://www.bestbuy.com/site/reviews/apple-airpods-pro-2-wireless-active-noise-cancelling-earbuds-with-hearing-aid-feature-white/6447382`
- Amazon:
  - any public product page or public review page that still exposes reviews without login
- Flipkart:
  - a direct `product-reviews` URL is usually the best starting point

## Secret Handling

- Keep real secrets in `.env`
- Commit `.env.example`
- `.env` is ignored by `.gitignore`

If a real `.env` was ever pushed, rotate the exposed key before recreating a remote repository.
