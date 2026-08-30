import asyncio
from typing import Optional
from loguru import logger
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

class AntiBotBrowser:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def get_page_content(self, url: str) -> Optional[str]:
        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await stealth_async(page)
        
        try:
            await asyncio.sleep(2.0)
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return await page.content()
        except Exception as e:
            logger.error(f"Anti-Bot navigation failed for {url}: {str(e)}")
            return None
        finally:
            await context.close()
