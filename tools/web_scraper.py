import logging
from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def get_rendered_html(url: str) -> dict:
    """Fetch complete page content using Playwright headless browser.
    
    Args:
        url: The URL to fetch
        
    Returns:
        dict with keys: html (page HTML), images (list of base64 encoded images), url
    """
    logger.info("=" * 80)
    logger.info("🌐 FETCHING RENDERED HTML")
    logger.info("=" * 80)
    logger.info(f"Target URL: {url}")
    
    try:
        with sync_playwright() as p:
            logger.info("🚀 Launching headless browser")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            logger.info(f"📄 Navigating to {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            logger.info("⏳ Waiting for content to load")
            page.wait_for_timeout(2000)
            
            logger.info("📝 Extracting HTML content")
            html_content = page.content()
            
            logger.info("🖼️  Extracting images")
            images = []
            img_elements = page.query_selector_all("img")
            
            for idx, img in enumerate(img_elements):
                try:
                    src = img.get_attribute("src")
                    if src and src.startswith("http"):
                        logger.info(f"   - Image {idx + 1}: {src[:60]}...")
                        screenshot = img.screenshot()
                        b64 = base64.b64encode(screenshot).decode()
                        images.append({"src": src, "base64": b64})
                except Exception as e:
                    logger.warning(f"   - Failed to capture image {idx + 1}: {str(e)}")
            
            browser.close()
            
            logger.info("=" * 80)
            logger.info("✓ PAGE FETCH COMPLETE")
            logger.info("=" * 80)
            logger.info(f"HTML size: {len(html_content)} chars")
            logger.info(f"Images captured: {len(images)}")
            logger.info("=" * 80)
            
            return {
                "html": html_content,
                "images": images,
                "url": url
            }
    
    except Exception as e:
        logger.error(f"❌ Error fetching page: {str(e)}")
        logger.exception(e)
        return {
            "error": str(e),
            "html": "",
            "images": [],
            "url": url
        }