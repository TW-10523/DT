# backend/models/document.py
# NEW: SQLAlchemy models for PostgreSQL + pgvector
# REUSED: conceptual schema from aviary (documents, chunks, embeddings) but converted to SQLAlchemy/pgvector

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import BYTEA
from app.core.db import Base
import datetime

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    title = Column(String, nullable=True)
    metadata = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    text = Column(Text, nullable=False)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    # embeddings stored in separate table or as vector column depending on pgvector setup
    embedding_id = Column(Integer, ForeignKey("embeddings.id"), nullable=True)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    # If using pgvector extension: vector column type; here we store as bytea for portability and
    # the services layer will write vector into pgvector column using raw SQL if needed.
    vector = Column(BYTEA, nullable=True)
    chunk = relationship("Chunk", back_populates="embedding")
