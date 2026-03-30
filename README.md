# Review Lens

Review Lens is a Python web application that scrapes customer reviews from an e-commerce product page, cleans and chunks the review text, summarizes each review with an OpenAI-compatible API, and presents the results in a browser UI that can be deployed to Vercel.

## What It Does

- Accepts a product page URL in a web form.
- Follows review links and paginated review pages when they are exposed in HTML.
- Extracts review text plus metadata such as rating, date, author, and title.
- Cleans noisy text and chunks long reviews before sending them to an LLM.
- Uses any OpenAI-compatible chat endpoint for concise summaries and sentiment labels.
- Lets users download the processed reviews as JSON or CSV from the browser.
- Includes retry logic, pacing delays, and clear failures for rate limits or anti-bot blocks.

## Stack

- Backend: Flask on Python
- Scraping: `requests`, `BeautifulSoup`, `urllib3` retry support
- LLM: `openai` Python SDK against OpenAI-compatible APIs
- Token-aware chunking: `tiktoken` with offline-safe fallback
- Deployment target: Vercel Python functions

## Project Layout

```text
api/
  index.py
review_scraper/
  adapters.py
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
  fixtures/
  test_pipeline.py
  test_web.py
vercel.json
requirements.txt
```

## Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Run Locally

```bash
python -m review_scraper
```

Then open `http://127.0.0.1:8000`.

If you already have `OPENAI_API_KEY` or `NVIDIA_API_KEY` in a root-level `.env` file, the app now loads it automatically on startup.

If you want to use an OpenAI-compatible provider instead of the default OpenAI environment variable, fill in the `Base URL`, `Model`, and either:

- set a server env var and leave the API key field empty
- paste a one-off API key into the form

## Example Provider Settings

For NVIDIA's compatible endpoint:

- Base URL: `https://integrate.api.nvidia.com/v1`
- Model: `deepseek-ai/deepseek-v3.1`
- API key env var: `NVIDIA_API_KEY`
- extra_body JSON: `{"chat_template_kwargs":{"thinking":true}}`

## Deploy To Vercel

1. Create a new Vercel project from this repository.
2. Add environment variables such as `OPENAI_API_KEY` or `NVIDIA_API_KEY` in the Vercel project settings.
3. Deploy as-is. The root route is rewritten to the Python function in `api/index.py`.

This repo includes:

- `api/index.py` as the Vercel Python entrypoint
- `vercel.json` to configure rewrites and function behavior

## Chosen Test URL

The interface is prefilled with this public review page:

`https://www.bestbuy.com/site/reviews/apple-airpods-pro-2-wireless-active-noise-cancelling-earbuds-with-hearing-aid-feature-white/6447382`

Retailer markup changes over time, and some sites block automation. If that page stops working, try another public product page that exposes reviews in HTML or JSON-LD.

## API Endpoint

The web UI posts to:

- `POST /api/reviews`

Expected JSON body fields include:

- `url`
- `skip_llm`
- `model`
- `base_url`
- `api_key`
- `api_key_env`
- `max_pages`
- `request_timeout`
- `extra_body_json`

The endpoint returns JSON by default and can return CSV when `response_format=csv`.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall review_scraper tests api
```

## Notes

- Vercel Python functions are still serverless functions, so long scrapes may approach execution limits on very large review sets.
- Some retailers, especially Amazon, often serve captcha or anti-bot pages. The scraper detects common block signals and returns an error instead of empty results.
- The app avoids relying on filesystem output during web requests so it remains safe for ephemeral deployments.
- The backend uses a simple `requests` fetch first and falls back to a minimal Playwright browser fetch when the site times out or blocks the HTTP request.
