"""messages_table

Revision ID: b55c7932cecb
Revises:
Create Date: 2025-04-20 15:24:25.695831

"""
from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b55c7932cecb"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable the pgvector extension if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create the messages table if it does not exist
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("uuid_work", sa.UUID, nullable=False),
        sa.Column("uuid_lead", sa.UUID, nullable=False),
        sa.Column(
            "timestamp", sa.TIMESTAMP, server_default=sa.func.now(), nullable=False
        ),
        sa.Column("role", sa.String(10)),
        sa.Column("message", sa.Text),
        sa.Column("embedding_vector", Vector(1536), nullable=False),
    )

    # Create indexes for uuid_work, uuid_lead, and both together
    op.create_index("ix_messages_uuid_work", "messages", ["uuid_work"])
    op.create_index("ix_messages_uuid_lead", "messages", ["uuid_lead"])
    op.create_index(
        "ix_messages_uuid_work_uuid_lead", "messages", ["uuid_work", "uuid_lead"]
    )


def downgrade() -> None:
    op.drop_table("messages")
    op.execute("DROP EXTENSION IF EXISTS vector")
