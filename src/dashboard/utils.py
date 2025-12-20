# src/dashboard/utils.py
import requests
import pandas as pd

# Backend base URL; dashboard assumes a local dev server. External deployments should override.
API_URL = "http://localhost:8000"


def check_api_status():
    """Probe health endpoint; used to gate UI actions without triggering inference."""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return True, response.json()
    except:
        pass
    return False, None


def predict_text(text: str):
    """Submit a single text to the API; returns JSON or an error payload."""
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
    Iterate rows and call predict per-item; simple but slow for large files due to per-request overhead.
    Suitable for demo-scale datasets; batch endpoints or async pipelines are recommended for production.
    """
    results = []
    for index, row in df.iterrows():
        text = str(row[text_col])
        res = predict_text(text)

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