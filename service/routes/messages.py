from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from db.models.messages import Message

messages_router = APIRouter()

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


# FastAPI endpoints
@messages_router.post("/persist_message")
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


@messages_router.post("/similarity_search")
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
