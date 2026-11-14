"""
Headless-browser fallback for pages that block plain HTTP clients (HTTP
403/429/503 from bot protection). Used by the news scraper only when a
normal request is refused.

Requires `playwright install chromium` once. If the browser isn't installed
the fallback disables itself and the caller simply skips that page.
"""
import asyncio
from typing import Optional

from loguru import logger

try:
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth
except ImportError:  # pragma: no cover - optional dependency
    async_playwright = None
    Stealth = None

# Status codes that usually mean "blocked as a bot" rather than "gone".
BLOCKED_STATUSES = {401, 403, 429, 503}


class AntiBotBrowser:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser = None

    async def __aenter__(self):
        if async_playwright is None:
            raise RuntimeError("playwright is not installed; pip install -r requirements.txt")
        self.playwright = await async_playwright().start()
        try:
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception:
            await self.playwright.stop()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def get_page_content(self, url: str) -> Optional[str]:
        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        # playwright-stealth 2.x API (1.x's `stealth_async` no longer exists).
        await Stealth().apply_stealth_async(page)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(1.0)  # let late scripts render the article body
            return await page.content()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Anti-Bot navigation failed for {url}: {e}")
            return None
        finally:
            await context.close()


class LazyAntiBotBrowser:
    """
    Starts the browser only the first time a page actually needs it, and
    disables itself for the rest of the run if it can't start (e.g. Chromium
    not installed), so one missing browser doesn't cost a launch per article.
    """

    def __init__(self) -> None:
        self._browser: Optional[AntiBotBrowser] = None
        self._unavailable = False
        self._lock = asyncio.Lock()

    async def get_page_content(self, url: str) -> Optional[str]:
        if self._unavailable:
            return None
        async with self._lock:
            if self._browser is None:
                try:
                    self._browser = await AntiBotBrowser().__aenter__()
                except Exception as e:  # noqa: BLE001
                    self._unavailable = True
                    logger.warning(
                        f"Anti-bot browser unavailable ({type(e).__name__}); "
                        "blocked pages will be skipped. Run `playwright install chromium` to enable it."
                    )
                    return None
        return await self._browser.get_page_content(url)

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.__aexit__(None, None, None)
            self._browser = None
