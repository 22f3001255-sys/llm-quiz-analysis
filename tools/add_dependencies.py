import logging
import subprocess
from langchain_core.tools import tool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def add_dependencies(packages: str) -> dict:
    """Install Python packages using uv add.
    
    Args:
        packages: Space-separated package names to install
        
    Returns:
        dict with keys: success, stdout, stderr
    """
    logger.info("=" * 80)
    logger.info("📦 INSTALLING DEPENDENCIES")
    logger.info("=" * 80)
    logger.info(f"Packages: {packages}")
    
    try:
        package_list = packages.split()
        logger.info(f"Installing {len(package_list)} package(s)")
        
        cmd = ["uv", "add"] + package_list
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("▶️  Executing")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        logger.info("=" * 80)
        logger.info("✓ INSTALLATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            logger.info("✅ Packages installed successfully")
        else:
            logger.error("❌ Installation failed")
        
        if result.stdout:
            logger.info("Stdout:")
            logger.info("-" * 40)
            for line in result.stdout.split('\n')[:20]:
                logger.info(f"  {line}")
            logger.info("-" * 40)
        
        if result.stderr:
            logger.warning("Stderr:")
            logger.warning("-" * 40)
            for line in result.stderr.split('\n')[:20]:
                logger.warning(f"  {line}")
            logger.warning("-" * 40)
        
        logger.info("=" * 80)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    except subprocess.TimeoutExpired:
        logger.error("❌ Installation timeout (120s)")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Installation timeout after 120 seconds"
        }
    
    except Exception as e:
        logger.error(f"❌ Error installing packages: {str(e)}")
        logger.exception(e)
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e)
        }