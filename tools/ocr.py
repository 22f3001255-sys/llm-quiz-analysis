import logging
from langchain_core.tools import tool
import pytesseract
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def ocr_image_tool(image_path: str) -> dict:
    """Extract text from an image using OCR (pytesseract).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with keys: text, success
    """
    logger.info("=" * 80)
    logger.info("🔍 EXTRACTING TEXT FROM IMAGE (OCR)")
    logger.info("=" * 80)
    logger.info(f"Image path: {image_path}")
    
    try:
        logger.info("📖 Opening image")
        image = Image.open(image_path)
        
        logger.info(f"Image size: {image.size}")
        logger.info(f"Image mode: {image.mode}")
        
        logger.info("🔍 Running OCR")
        text = pytesseract.image_to_string(image)
        
        logger.info("=" * 80)
        logger.info("✓ OCR COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Extracted text length: {len(text)} chars")
        
        if text.strip():
            logger.info("Text preview:")
            logger.info("-" * 40)
            preview_lines = text.split('\n')[:10]
            for line in preview_lines:
                logger.info(f"  {line}")
            if len(text.split('\n')) > 10:
                logger.info("  ...")
            logger.info("-" * 40)
        else:
            logger.warning("⚠️ No text extracted from image")
        
        logger.info("=" * 80)
        
        return {
            "text": text,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"❌ Error extracting text: {str(e)}")
        logger.exception(e)
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }