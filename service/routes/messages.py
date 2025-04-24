from typing import Optional

from fastapi import APIRouter, HTTPException
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pydantic import BaseModel

from db.models.messages import Message

messages_router = APIRouter(prefix="/v1")

# Initialize database
message_db = Message()

# Initialize embeddings model
embeddings = OpenAIEmbeddings()


# Pydantic models for request bodies
class PersistMessageRequest(BaseModel):
    uuid_work: str
    uuid_lead: str
    role: str
    text: str
    embeddings: Optional[list] = None


class RetrieveMemoryRequest(BaseModel):
    message_text: str
    similarity_search_threshold: float
    similarity_search_limit: int
    uuid_work: str
    uuid_lead: str


# FastAPI endpoints
@messages_router.post("/persist_message")
async def persist_message_endpoint(request: PersistMessageRequest):
    try:
        # Check if embeddings are provided, if not generate them
        if not request.embeddings:
            request.embeddings = embeddings.embed_query(request.text)

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


@messages_router.post("/retrieve_memory")
async def retrieve_memory_endpoint(request: RetrieveMemoryRequest):
    try:
        # Generate embeddings for the message
        message_embedding = embeddings.embed_query(request.message_text)
        role = "user"

        # Perform similarity search
        results = await message_db.similarity_search(
            message_embedding,
            request.similarity_search_threshold,
            request.similarity_search_limit,
            request.uuid_work,
            request.uuid_lead,
        )

        # Persist the message
        await message_db.persist_message(
            request.uuid_work,
            request.uuid_lead,
            role,
            request.message_text,
            message_embedding,
        )

        return {"results": results}
    except Exception as e:
        logger.error(f"Error in retrieve_memory: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
