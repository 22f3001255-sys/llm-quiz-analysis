import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from datetime import datetime

from agent import run_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SolveRequest(BaseModel):
    url: str
    secret: str

@app.get("/healthz")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/solve")
async def solve_quiz(request: SolveRequest, background_tasks: BackgroundTasks):
    logger.info("=" * 80)
    logger.info("📨 RECEIVED SOLVE REQUEST")
    logger.info("=" * 80)
    logger.info(f"URL: {request.url}")
    logger.info(f"Secret provided: {'✓' if request.secret else '✗'}")
    
    expected_secret = os.getenv("SECRET")
    if not expected_secret:
        logger.error("❌ SECRET not configured in environment")
        raise HTTPException(status_code=500, detail="Server configuration error")
    
    if request.secret != expected_secret:
        logger.warning("❌ Invalid secret provided")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info("✓ Authentication successful")
    logger.info("🚀 Starting agent in background task")
    background_tasks.add_task(run_agent, request.url)
    
    logger.info("✓ Background task queued")
    logger.info("=" * 80)
    return {"status": "started", "url": request.url}

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🎯 QUIZ SOLVING AGENT SERVER")
    logger.info("=" * 80)
    logger.info(f"Starting server on 0.0.0.0:7860")
    logger.info(f"Email configured: {os.getenv('EMAIL', 'NOT SET')}")
    logger.info(f"Secret configured: {'✓' if os.getenv('SECRET') else '✗'}")
    logger.info("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=7860)