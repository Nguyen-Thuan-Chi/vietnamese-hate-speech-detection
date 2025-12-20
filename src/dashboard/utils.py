# src/dashboard/utils.py
import requests
import pandas as pd

# Địa chỉ API Backend (Mặc định chạy local)
API_URL = "http://localhost:8000"


def check_api_status():
    """Ping xem Server sống hay chết"""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return True, response.json()
    except:
        pass
    return False, None


def predict_text(text: str):
    """Gửi 1 câu sang API để soi"""
    try:
        payload = {"text": text}
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def predict_csv(df: pd.DataFrame, text_col: str):
    """
    Chạy vòng lặp quét cả file Excel/CSV.
    Lưu ý: Cách này hơi chậm nếu file lớn (gọi API từng dòng).
    Nhưng với Demo đồ án thì OK.
    """
    results = []
    # Tạo thanh loading giả lập trong Streamlit sau
    for index, row in df.iterrows():
        text = str(row[text_col])
        res = predict_text(text)

        # Lấy kết quả
        if "error" not in res:
            results.append({
                "Original Text": text,
                "Clean Text": res.get("clean_text", ""),
                "Label": res.get("label", "UNKNOWN"),
                "Confidence": res.get("confidence", "0%")
            })
        else:
            results.append({
                "Original Text": text,
                "Label": "ERROR",
                "Confidence": "0%"
            })

    return pd.DataFrame(results)