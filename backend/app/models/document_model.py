# core/db.py
from sqlalchemy import (
    Column, Integer, BigInteger, Text, TIMESTAMP, JSON, ForeignKey, String
)
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import VECTOR  # available via sqlalchemy-pgvector or custom type

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_name = Column(Text, nullable=False, index=True)
    filename = Column(Text)
    uploaded_by = Column(Text)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id = Column(BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", uselist=False, back_populates="chunk", cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id = Column(BigInteger, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    # The VECTOR type requires sqlalchemy-pgvector or a custom type registration.
    embedding = Column(VECTOR(1536))  # change 1536 to your dimension
    collection_name = Column(Text, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    chunk = relationship("Chunk", back_populates="embedding")
