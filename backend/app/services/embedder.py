from sentence_transformers import SentenceTransformer
from app.core.config import settings
import jaconv
import torch

# Use the correct embedding model (BGE-M3)
model_name = settings.EMBEDDING_MODEL  # "BAAI/bge-m3"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    model_name,
    device=device,
    trust_remote_code=True  # Required for BGE-M3
)

# テキスト正規化
def process_text(text):
    text = jaconv.z2h(text, kana=False, digit=True, ascii=True)
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    return text

def embed_text(text: str):
    """
    ベクトル化された 1024 次元 BGE-M3 埋め込みを返す
    """
    text = process_text(text)
    emb = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=1
    )
    return emb.tolist()
