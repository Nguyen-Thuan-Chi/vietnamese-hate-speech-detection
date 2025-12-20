from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch

# Lưu ý: Import đúng đường dẫn file predictor của bạn
# Nếu file nằm ở src/services/predictor.py thì sửa lại import bên dưới
from src.services.predictor import HateSpeechPredictor

# ====== PATH (Code của bạn - Rất chuẩn) ======
# src/api/server.py -> parents[0]=api -> parents[1]=src -> parents[2]=root
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "phobert_epoch_3.pth"

print(f"--> [DEBUG] Đang tìm model tại: {MODEL_PATH}")

if not MODEL_PATH.exists():
    # Dùng RuntimeError để dừng server ngay lập tức nếu thiếu model
    raise RuntimeError(f"❌ Không tìm thấy model tại: {MODEL_PATH}")

# ====== DEVICE ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== LOAD MODEL ======
try:
    # Load model với n_classes=2 (Binary)
    # Lưu ý: Class HateSpeechPredictor phải khớp với code cũ
    predictor = HateSpeechPredictor(str(MODEL_PATH), device=device)
    print("--> [SERVER] Model đã sẵn sàng!")
except Exception as e:
    raise RuntimeError(f"❌ Không load được model: {e}")

# ====== API DEFINITION ======
app = FastAPI(title="Hate Speech Detection API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: str  # Để String vì bên predictor trả về "99.50%"
    clean_text: str  # Thêm cái này để debug xem nó clean đúng không


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Hàm này trả về Dict: {'label': 'TOXIC', 'confidence': '99.5%', ...}
        result = predictor.predict(req.text)

        return PredictResponse(
            label=result['label'],
            confidence=result['confidence'],
            clean_text=result['text_clean']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chạy trực tiếp để test (nếu cần)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)