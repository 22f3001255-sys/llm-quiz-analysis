import logging
import base64
import uuid
from langchain_core.tools import tool
from shared import BASE64_STORE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def encode_image_to_base64(image_path: str) -> dict:
    """Encode an image to base64 and store in BASE64_STORE.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with keys: placeholder (BASE64_KEY:uuid), uuid, success
    """
    logger.info("=" * 80)
    logger.info("🖼️  ENCODING IMAGE TO BASE64")
    logger.info("=" * 80)
    logger.info(f"Image path: {image_path}")
    
    try:
        logger.info("📖 Reading image file")
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        logger.info(f"📦 Image size: {len(image_data)} bytes")
        
        logger.info("🔐 Encoding to base64")
        b64_string = base64.b64encode(image_data).decode('utf-8')
        
        logger.info(f"✓ Base64 length: {len(b64_string)} chars")
        
        key = str(uuid.uuid4())
        BASE64_STORE[key] = b64_string
        
        placeholder = f"BASE64_KEY:{key}"
        
        logger.info("=" * 80)
        logger.info("✓ ENCODING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"UUID: {key}")
        logger.info(f"Placeholder: {placeholder}")
        logger.info(f"Stored in BASE64_STORE (total entries: {len(BASE64_STORE)})")
        logger.info("=" * 80)
        
        return {
            "placeholder": placeholder,
            "uuid": key,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"❌ Error encoding image: {str(e)}")
        logger.exception(e)
        return {
            "placeholder": "",
            "uuid": "",
            "success": False,
            "error": str(e)
        }