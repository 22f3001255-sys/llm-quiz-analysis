import logging
import os
import time
import json
import requests
from langchain_core.tools import tool
from shared import BASE64_STORE
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Rate limiting
request_times = []
MAX_REQUESTS = 4
TIME_WINDOW = 60

def check_rate_limit():
    """Check if rate limit is exceeded"""
    global request_times
    now = time.time()
    request_times = [t for t in request_times if now - t < TIME_WINDOW]
    
    if len(request_times) >= MAX_REQUESTS:
        wait_time = TIME_WINDOW - (now - request_times[0])
        if wait_time > 0:
            logger.warning(f"⏳ Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            request_times = []
    
    request_times.append(now)

@tool
def post_request(url: str, payload: dict, max_retries: int = 4) -> dict:
    """Submit answer via POST request. Handles BASE64_KEY placeholders and retries.
    
    Args:
        url: Submit endpoint URL
        payload: Dictionary containing answer data
        max_retries: Maximum retry attempts
        
    Returns:
        dict with response data including correct, reason, next_url
    """
    logger.info("\n" + "=" * 80)
    logger.info("📤 SUBMITTING ANSWER")
    logger.info("=" * 80)
    logger.info(f"Submit URL: {url}")
    
    # Ensure email and secret are included
    email = os.getenv("EMAIL")
    secret = os.getenv("SECRET")
    
    if "email" not in payload:
        payload["email"] = email
    if "secret" not in payload:
        payload["secret"] = secret
    
    # Replace BASE64_KEY placeholders
    payload_str = json.dumps(payload)
    for key, value in BASE64_STORE.items():
        placeholder = f"BASE64_KEY:{key}"
        if placeholder in payload_str:
            logger.info(f"🔄 Replacing {placeholder} with base64 data")
            payload_str = payload_str.replace(placeholder, value)
    
    payload = json.loads(payload_str)
    
    logger.info("Payload:")
    logger.info("-" * 40)
    # Don't log full base64 or secret
    safe_payload = payload.copy()
    if "secret" in safe_payload:
        safe_payload["secret"] = "***"
    for k, v in safe_payload.items():
        if isinstance(v, str) and len(v) > 100:
            safe_payload[k] = v[:50] + "..." + v[-20:]
    logger.info(json.dumps(safe_payload, indent=2))
    logger.info("-" * 40)
    
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        
        logger.info(f"🔄 Attempt {attempt}/{max_retries}")
        
        try:
            check_rate_limit()
            
            logger.info("🌐 Sending POST request")
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            logger.info(f"✓ Response status: {response.status_code}")
            
            try:
                data = response.json()
            except:
                data = {"text": response.text}
            
            logger.info("=" * 80)
            logger.info("📨 RESPONSE DATA")
            logger.info("=" * 80)
            logger.info(json.dumps(data, indent=2))
            logger.info("=" * 80)
            
            if response.status_code == 200:
                correct = data.get("correct", False)
                reason = data.get("reason", "")
                next_url = data.get("url") or data.get("next_url")
                
                if correct:
                    logger.info("✅ ANSWER CORRECT!")
                else:
                    logger.warning("❌ ANSWER INCORRECT")
                    logger.warning(f"Reason: {reason}")
                
                if next_url:
                    logger.info(f"🔗 Next URL: {next_url}")
                else:
                    logger.info("🏁 No next URL - quiz chain complete")
                
                return {
                    "status_code": response.status_code,
                    "correct": correct,
                    "reason": reason,
                    "next_url": next_url,
                    "data": data,
                    "attempt": attempt
                }
            else:
                logger.warning(f"⚠️ Non-200 status: {response.status_code}")
                if attempt < max_retries:
                    logger.info(f"⏳ Retrying in 2s")
                    time.sleep(2)
                else:
                    logger.error("❌ Max retries exceeded")
                    return {
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}",
                        "data": data,
                        "attempt": attempt
                    }
        
        except Exception as e:
            logger.error(f"❌ Request error: {str(e)}")
            if attempt < max_retries:
                logger.info(f"⏳ Retrying in 2s")
                time.sleep(2)
            else:
                logger.error("❌ Max retries exceeded")
                logger.exception(e)
                return {
                    "error": str(e),
                    "attempt": attempt
                }
    
    return {"error": "Max retries exceeded", "attempt": max_retries}