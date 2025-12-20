# src/api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn

# Import class dự đoán
from src.services.predictor import HateSpeechPredictor

# ====== 1. PATH CONFIG ======
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "phobert_epoch_3.pth"

print(f"--> [DEBUG] Đang tìm model tại: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise RuntimeError(f"❌ Không tìm thấy model tại: {MODEL_PATH}")

# ====== 2. DEVICE & MODEL LOADING ======
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load model Binary (n_classes=2)
    predictor = HateSpeechPredictor(str(MODEL_PATH), device=device)
    print("--> [SERVER] Model đã sẵn sàng!")
except Exception as e:
    raise RuntimeError(f"❌ Không load được model: {e}")

# ====== 3. API SETUP ======
app = FastAPI(title="Hate Speech Detection API")

# Schemas
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: str
    clean_text: str

# --- [QUAN TRỌNG] Endpoint kiểm tra sức khỏe (Health Check) ---
# Dashboard sẽ gọi vào đây để biết server còn sống hay không
@app.get("/")
def health_check():
    return {"status": "healthy", "device": device}

# --- Endpoint dự đoán chính ---
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predictor.predict(req.text)
        return PredictResponse(
            label=result['label'],
            confidence=result['confidence'],
            clean_text=result['text_clean']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)