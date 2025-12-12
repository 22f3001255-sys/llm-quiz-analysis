import logging
import subprocess
import tempfile
import os
from langchain_core.tools import tool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def run_code(code: str) -> dict:
    """Execute Python code using uv run in a subprocess.
    
    Args:
        code: Python code to execute
        
    Returns:
        dict with keys: stdout, stderr, return_code
    """
    logger.info("=" * 80)
    logger.info("🐍 EXECUTING PYTHON CODE")
    logger.info("=" * 80)
    logger.info(f"Code length: {len(code)} chars")
    logger.info("Code preview:")
    logger.info("-" * 40)
    preview_lines = code.split('\n')[:5]
    for line in preview_lines:
        logger.info(f"  {line}")
    if len(code.split('\n')) > 5:
        logger.info("  ...")
    logger.info("-" * 40)
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        logger.info(f"📝 Code written to: {temp_file}")
        logger.info("▶️  Executing with uv run")
        
        result = subprocess.run(
            ["uv", "run", "--no-project", temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        os.unlink(temp_file)
        
        logger.info("=" * 80)
        logger.info("✓ CODE EXECUTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Return code: {result.returncode}")
        logger.info(f"Stdout length: {len(result.stdout)} chars")
        logger.info(f"Stderr length: {len(result.stderr)} chars")
        
        if result.stdout:
            logger.info("Stdout preview:")
            logger.info("-" * 40)
            for line in result.stdout.split('\n')[:10]:
                logger.info(f"  {line}")
            logger.info("-" * 40)
        
        if result.stderr:
            logger.warning("Stderr:")
            logger.warning("-" * 40)
            for line in result.stderr.split('\n')[:10]:
                logger.warning(f"  {line}")
            logger.warning("-" * 40)
        
        logger.info("=" * 80)
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    except subprocess.TimeoutExpired:
        logger.error("❌ Code execution timeout (30s)")
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {
            "stdout": "",
            "stderr": "Execution timeout after 30 seconds",
            "return_code": -1
        }
    
    except Exception as e:
        logger.error(f"❌ Error executing code: {str(e)}")
        logger.exception(e)
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.unlink(temp_file)
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }