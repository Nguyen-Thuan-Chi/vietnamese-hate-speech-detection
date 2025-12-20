# src/api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn

from src.services.predictor import HateSpeechPredictor

# Resolve model checkpoints relative to repo root; fail fast if missing to avoid serving partial functionality
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "phobert_epoch_3.pth"

print(f"--> [DEBUG] Đang tìm model tại: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise RuntimeError(f"❌ Không tìm thấy model tại: {MODEL_PATH}")

# Choose device at startup; inference latency depends on this selection, but correctness should not
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Predictor encapsulates preprocessing + model; constructed once to avoid per-request overhead
    predictor = HateSpeechPredictor(str(MODEL_PATH), device=device)
    print("--> [SERVER] Model đã sẵn sàng!")
except Exception as e:
    raise RuntimeError(f"❌ Không load được model: {e}")

# API surface kept minimal: health and single prediction endpoint for synchronous use cases
app = FastAPI(title="Hate Speech Detection API")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: str
    clean_text: str

# Health endpoint used by dashboards/probes; indicates device and readiness without triggering inference
@app.get("/")
def health_check():
    return {"status": "healthy", "device": device}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Basic input validation to avoid degenerate requests and excessive payloads
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung, không được để trống.")

    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Nội dung quá dài (tối đa 2000 ký tự).")

    try:
        result = predictor.predict(req.text)
        return PredictResponse(
            label=result['label'],
            confidence=result['confidence'],
            clean_text=result['text_clean']
        )
    except Exception as e:
        # Surface internal errors as 500; detailed logging should be added in production
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)