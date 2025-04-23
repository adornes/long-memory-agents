import os
from datetime import datetime

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class Database:
    def __init__(self):
        self.pool = AsyncConnectionPool(
            conninfo=os.getenv("DATABASE_URL"),
            max_size=20,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

    async def persist_message(self, uuid_work, uuid_lead, role, text, embeddings):
        current_timestamp = datetime.now()
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO chat (uuid_work, uuid_lead, timestamp, role, message, embedding_vector)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (uuid_work, uuid_lead, current_timestamp, role, text, embeddings),
            )

    async def similarity_search(
        self,
        message_embedding,
        similarity_search_threshold,
        similarity_search_limit,
        uuid_work,
        uuid_lead,
    ):
        async with self.pool.connection() as conn:
            similarity_search = await conn.execute(
                """
                WITH ranked_messages AS (
                    SELECT
                        message,
                        1 - (embedding_vector <=> %s::vector) AS cosine_similarity,
                        ROW_NUMBER() OVER (
                            PARTITION BY message
                            ORDER BY 1 - (embedding_vector <=> %s::vector) DESC
                        ) as rn
                    FROM chat
                    WHERE 1 - (embedding_vector <=> %s::vector) >= %s
                    AND uuid_work = %s
                    AND uuid_lead = %s
                )
                SELECT message, cosine_similarity
                FROM ranked_messages
                WHERE rn = 1
                ORDER BY cosine_similarity DESC
                LIMIT %s;
                """,
                (
                    message_embedding,
                    message_embedding,
                    message_embedding,
                    similarity_search_threshold,
                    uuid_work,
                    uuid_lead,
                    similarity_search_limit,
                ),
            )
            return await similarity_search.fetchall()
