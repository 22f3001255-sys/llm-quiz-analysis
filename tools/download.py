import logging
import os
import requests
from langchain_core.tools import tool
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def download_file(url: str, filename: str = None) -> dict:
    """Download a file from URL and save to LLMFiles directory.
    
    Args:
        url: URL to download from
        filename: Optional custom filename. If not provided, extracted from URL
        
    Returns:
        dict with keys: filepath, size, filename, success
    """
    logger.info("=" * 80)
    logger.info("📥 DOWNLOADING FILE")
    logger.info("=" * 80)
    logger.info(f"URL: {url}")
    
    try:
        os.makedirs("LLMFiles", exist_ok=True)
        
        if not filename:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or "downloaded_file"
        
        filepath = os.path.join("LLMFiles", filename)
        logger.info(f"Target path: {filepath}")
        
        logger.info("🌐 Sending request")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        logger.info(f"✓ Response received: {response.status_code}")
        logger.info(f"📦 Content size: {len(response.content)} bytes")
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(filepath)
        
        logger.info("=" * 80)
        logger.info("✓ FILE DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Saved to: {filepath}")
        logger.info(f"File size: {file_size} bytes")
        logger.info("=" * 80)
        
        return {
            "filepath": filepath,
            "size": file_size,
            "filename": filename,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"❌ Error downloading file: {str(e)}")
        logger.exception(e)
        return {
            "filepath": "",
            "size": 0,
            "filename": filename or "",
            "success": False,
            "error": str(e)
        }