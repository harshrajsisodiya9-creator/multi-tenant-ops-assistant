from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Computed, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType


class TSVector(UserDefinedType):
    cache_ok = True

    def get_col_spec(self):
        return "TSVECTOR"

from app.database import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(unique=True)
    api_key_hash: Mapped[str]

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="tenant")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_document_id",
            "chunk_index",
            name="uq_chunks_tenant_document_chunk",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    source_document_id: Mapped[str] = mapped_column(String, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    content_tsv = Column("content_tsv", TSVector(), Computed("to_tsvector('english', content)", persisted=True))

    tenant: Mapped[Tenant] = relationship(back_populates="chunks")
