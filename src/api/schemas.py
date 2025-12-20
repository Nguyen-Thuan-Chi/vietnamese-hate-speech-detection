# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

# 1. Định nghĩa cái khuôn cho Dữ liệu Gửi lên (Request)
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Mày ngu vãi")
    # min_length=1: Không được gửi chuỗi rỗng
    # example: Để hiện ví dụ trên giao diện Swagger UI tự động

# 2. Định nghĩa cái khuôn cho Dữ liệu Trả về (Response)
class PredictResponse(BaseModel):
    input_text: str
    clean_text: str
    label: str       # "CLEAN" hoặc "TOXIC"
    confidence: str  # Ví dụ: "99.50%"
    processing_time: float # Thời gian xử lý (tính bằng giây) - Để khoe tốc độ

    class Config:
        json_schema_extra = {
            "example": {
                "input_text": "Mày ngu vãi",
                "clean_text": "mày ngu vãi",
                "label": "TOXIC",
                "confidence": "99.68%",
                "processing_time": 0.045
            }
        }