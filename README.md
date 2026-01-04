# Vietnamese Hate Speech Detection System

An academic project for Vietnamese hate speech detection using a fine-tuned PhoBERT model.

> **📖 Full documentation:** [README_FULL.md](README_FULL.md)

---

## Overview

| Attribute | Value |
|-----------|-------|
| Task | Binary classification (TOXIC / CLEAN) |
| Model | PhoBERT (`vinai/phobert-base-v2`) |
| Prediction Level | **Sentence-level only** |
| UI | Streamlit dashboard |
| API | FastAPI |

> ⚠️ **This model does NOT perform span-level or token-level prediction.** It classifies entire sentences.

---

## Quick Start

### 1. Setup

```bash
git clone <repository-url>
cd ViHOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model

Place the checkpoint file in `models/phobert_epoch_3.pth`.

Model weights are **not included** in this repository.

### 3. Run

**Streamlit Dashboard:**
```bash
streamlit run src/dashboard/app.py
```
→ http://localhost:8501

**FastAPI Backend:**
```bash
uvicorn src.api.server:app --reload
```
→ http://localhost:8000/docs

---

## Key Features

- **Sentence classification:** TOXIC or CLEAN with confidence score
- **Keyword highlighting:** Heuristic fallback (not model output)
- **LIME explanations:** Post-hoc interpretability for TOXIC predictions
- **Batch processing:** Upload CSV/XLSX for bulk analysis
- **Feedback collection:** User corrections saved to CSV

---

## Important Limitations

| Limitation | Description |
|------------|-------------|
| Sentence-level only | No word or span-level predictions |
| Keyword highlighting | Heuristic, not model-based |
| LIME | Post-hoc explanation, not model output |
| Vietnamese only | Not trained on other languages |
| No auto-retraining | Feedback is stored but not used automatically |

---

## API

**POST /predict**

```
// Request
{"text": "Nội dung cần kiểm tra"}

// Response
{"label": "TOXIC", "confidence": "99.69%", "clean_text": "..."}
```

---

## Citation

```bibtex
@inproceedings{hoang-etal-2023-vihos,
  title = {ViHOS: Vietnamese Hate and Offensive Spans Detection},
  author = {Hoang, Phu Gia and Luu, Canh Duc and Tran, Khanh Quoc and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle = {EACL 2023},
  year = {2023}
}
```

---

## License

See [LICENSE](LICENSE) for details.
