import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from db.models.messages import Message

# Load environment variables
load_dotenv()

# Get environment variables with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
)
logger.add(LOG_FILE, rotation="10 MB", retention="1 week", level=LOG_LEVEL)

app = FastAPI(
    title="Long Memory Agents API",
    description="API for managing long memory agents",
    version="0.1.0",
    debug=DEBUG,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
message_db = Message()


# Pydantic models for request bodies
class PersistMessageRequest(BaseModel):
    uuid_work: str
    uuid_lead: str
    role: str
    text: str
    embeddings: list


class SimilaritySearchRequest(BaseModel):
    message_embedding: list
    similarity_search_threshold: float
    similarity_search_limit: int
    uuid_work: str
    uuid_lead: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


# FastAPI endpoints
@app.post("/persist_message")
async def persist_message_endpoint(request: PersistMessageRequest):
    try:
        await message_db.persist_message(
            request.uuid_work,
            request.uuid_lead,
            request.role,
            request.text,
            request.embeddings,
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error persisting message: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/similarity_search")
async def similarity_search_endpoint(request: SimilaritySearchRequest):
    try:
        results = await message_db.similarity_search(
            request.message_embedding,
            request.similarity_search_threshold,
            request.similarity_search_limit,
            request.uuid_work,
            request.uuid_lead,
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Long Memory Agents API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
