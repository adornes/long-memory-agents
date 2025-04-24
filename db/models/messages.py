from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Float, String, literal, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from logger_config import logger

from . import Base, SessionLocal


class MessageORM(Base):
    __tablename__ = "messages"

    uuid_work = Column(UUID(as_uuid=True), primary_key=True)
    uuid_lead = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    role = Column(String, nullable=False)
    message = Column(String, nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)


class Message:
    def __init__(self):
        self.session = SessionLocal()

    def persist_message(self, uuid_work, uuid_lead, role, text, embeddings):
        logger.info(
            f"Persisting message for work UUID: {uuid_work} and lead UUID: {uuid_lead}"
        )
        current_timestamp = datetime.now()
        new_message = MessageORM(
            uuid_work=uuid_work,
            uuid_lead=uuid_lead,
            timestamp=current_timestamp,
            role=role,
            message=text,
            embedding_vector=embeddings,
        )
        self.session.add(new_message)
        self.session.commit()

    def similarity_search(
        self,
        message_embedding,
        similarity_search_threshold,
        similarity_search_limit,
        uuid_work,
        uuid_lead,
    ):
        logger.info(
            f"Performing similarity search for work UUID: {uuid_work} and lead UUID: {uuid_lead}"
        )
        query = self.session.query(
            MessageORM.message,
            (
                1
                - MessageORM.embedding_vector.op("<=>")(
                    literal(message_embedding, Vector(1536))
                )
            ).label("cosine_similarity"),
        )
        query = (
            query.filter(
                (
                    1
                    - MessageORM.embedding_vector.op("<=>")(
                        literal(message_embedding, Vector(1536))
                    )
                )
                >= similarity_search_threshold,
                MessageORM.uuid_work == uuid_work,
                MessageORM.uuid_lead == uuid_lead,
            )
            .order_by(text("cosine_similarity DESC"))
            .limit(similarity_search_limit)
        )
        return query.all()
