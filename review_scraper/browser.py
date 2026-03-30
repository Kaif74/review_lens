from __future__ import annotations

import contextlib

from .models import ScraperConfig

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


class BrowserFetchError(RuntimeError):
    """Raised when the browser fallback is unavailable or fails."""


class BrowserFetcher:
    def __init__(self, config: ScraperConfig):
        self.config = config

    def fetch_html(self, url: str, user_agent: str) -> str:
        if sync_playwright is None:
            raise BrowserFetchError(
                "Browser fallback requires Playwright. Install it with "
                "'pip install playwright' and then run 'python -m playwright install chromium'."
            )

        try:
            with sync_playwright() as playwright:
                errors: list[str] = []
                for channel in ["msedge", "chrome", None]:
                    browser = None
                    context = None
                    try:
                        launch_kwargs = {
                            "headless": self.config.browser_headless,
                            "args": [
                                "--disable-http2",
                                "--disable-features=IsolateOrigins,site-per-process",
                                "--disable-blink-features=AutomationControlled",
                            ],
                        }
                        if channel is not None:
                            launch_kwargs["channel"] = channel

                        browser = playwright.chromium.launch(**launch_kwargs)
                        context = browser.new_context(
                            user_agent=user_agent,
                            locale="en-US",
                            viewport={"width": 1440, "height": 1200},
                            ignore_https_errors=True,
                        )
                        page = context.new_page()
                        page.route("**/*", _route_handler)
                        page.goto(url, wait_until="commit", timeout=self.config.browser_timeout_ms)
                        page.wait_for_timeout(2500)
                        return page.content()
                    except Exception as exc:
                        label = channel or "bundled-chromium"
                        errors.append(f"{label}: {exc}")
                    finally:
                        with contextlib.suppress(Exception):
                            if context is not None:
                                context.close()
                        with contextlib.suppress(Exception):
                            if browser is not None:
                                browser.close()
                raise BrowserFetchError(f"Browser attempts failed for {url}: {' | '.join(errors)}")
        except Exception as exc:
            raise BrowserFetchError(f"Browser fallback failed for {url}: {exc}") from exc


def _route_handler(route) -> None:
    request = route.request
    if request.resource_type in {"image", "media", "font", "stylesheet"}:
        route.abort()
        return
    route.continue_()
