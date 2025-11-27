from transformers import AutoTokenizer, AutoModel
from core.config import settings
import torch
import jaconv

# Load model and tokenizer for manual embedding
model_name = settings.EMBEDDING_MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

if device == "cuda":
    model.half()

# テキスト正規化
def process_text(text):
    # 全角英数字・記号を半角に変換
    text = jaconv.z2h(text, kana=False, digit=True, ascii=True)
    
    # 空白と改行を削除
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    
    return text

def embed_text(text: str):
    """
    テキストをベクトル化します。
    Args:
        text (str): ベクトル化するテキスト。
    Returns:
        list: ベクトル化されたリスト。
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()
    return embedding
