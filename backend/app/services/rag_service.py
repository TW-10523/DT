# app/services/rag_service.py
from app.repositories.postgres_repository import PostgresRepository
from app.services.inference import generate

repo = PostgresRepository()  # reuse your repo

def answer_with_rag(user_query, k=4):
    # 1) embed query (use embedder.py)
    hits = repo.semantic_search(user_query, top_k=k)  # implement this in postgres_repository
    context = "\n\n---\n\n".join([h["text"] for h in hits])
    prompt = f"Use the context below to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_query}\n\nAnswer concisely."
    return generate(prompt, max_new_tokens=256)
