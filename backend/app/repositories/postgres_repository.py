# repositories/postgres_repository.py
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.models.document import Document, Chunk, Embedding
from app.models.document import Base
from app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.DATABASE_URL, future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

class PostgresRepository:
    def __init__(self):
        # Make sure tables are created (or use Alembic migrations)
        Base.metadata.create_all(bind=engine)

    def create_document(self, collection_name: str, filename: str, uploaded_by: str = None):
        with SessionLocal() as session:
            doc = Document(collection_name=collection_name, filename=filename, uploaded_by=uploaded_by)
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    def add_chunk_with_embedding(self, document_id: int, chunk_index: int, text: str, embedding_vector, collection_name: str, metadata: dict = None):
        with SessionLocal() as session:
            chunk = Chunk(document_id=document_id, chunk_index=chunk_index, text=text, metadata=metadata or {})
            session.add(chunk)
            session.flush()  # get chunk.id
            emb = Embedding(chunk_id=chunk.id, embedding=embedding_vector, collection_name=collection_name)
            session.add(emb)
            session.commit()
            session.refresh(emb)
            return emb

    def delete_collection(self, collection_name: str):
        with SessionLocal() as session:
            # delete documents and cascading chunks/embeddings
            session.query(Document).filter(Document.collection_name == collection_name).delete(synchronize_session=False)
            session.commit()
            return {"deleted_collection": collection_name}

    def get_chunks_by_ids(self, chunk_ids: list):
        with SessionLocal() as session:
            return session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()

    def nearest_neighbors(self, query_vector, collection_name: str, top_k: int = 10):
        """
        Returns top_k rows (chunk_id, score, text, metadata)
        Uses pgvector '<->' operator for distance (euclidean). For cosine, transform accordingly.
        """
        # query_vector must be a Python list/tuple. We'll convert to SQL literal.
        with SessionLocal() as session:
            # Bind vector literal safely
            # Postgres accepts string like '[0.1, 0.2, ...]'::vector
            vec_literal = "[" + ",".join(map(str, query_vector)) + "]"
            sql = text(f"""
                SELECT e.chunk_id, e.embedding <-> :q AS distance, c.text, c.metadata
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                WHERE e.collection_name = :collection_name
                ORDER BY e.embedding <-> :q
                LIMIT :k
            """)
            res = session.execute(sql, {"q": vec_literal, "collection_name": collection_name, "k": top_k})
            rows = []
            for row in res:
                rows.append({
                    "chunk_id": row.chunk_id,
                    "distance": float(row.distance),
                    "text": row.text,
                    "metadata": row.metadata
                })
            return rows
