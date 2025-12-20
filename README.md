# Vietnamese Hate Speech Detection System

## Overview
This repository contains an academic application project for detecting Vietnamese hate and offensive text. The system uses a fine-tuned PhoBERT model for inference, exposed via a FastAPI backend and an interactive Streamlit dashboard.

- Model: PhoBERT (fine-tuned, inference-only)
- Backend: FastAPI + Uvicorn
- Dashboard: Streamlit
- Task: Binary classification (TOXIC vs CLEAN) with confidence scores

## System Architecture
User → Streamlit Dashboard → FastAPI (Uvicorn) → PhoBERT Model

---

## Installation

### Prerequisites
- Python 3.8+
- Optional: CUDA-compatible GPU for faster inference

### Setup
Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1) Start the Backend API
This service loads the model and exposes prediction endpoints.

```bash
uvicorn src.api.server:app --reload
```

- API URL: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 2) Start the Dashboard
In a separate terminal, run:

```bash
streamlit run src/dashboard/app.py
```

- Dashboard URL: http://localhost:8501

---

## API Reference

### POST /predict

Request body:

```json
{
  "text": "Nội dung cần kiểm tra"
}
```

Response body:

```json
{
  "label": "TOXIC",
  "confidence": "99.69%",
  "clean_text": "nội dung cần kiểm tra"
}
```

---

## Dataset & Acknowledgement
This project is inspired by the ViHOS dataset (Vietnamese Hate and Offensive Spans Detection).

- ViHOS is a span-level dataset.
- This project adapts the data for text-level binary classification suitable for moderation systems.

The dataset examples may contain offensive language; they are used for research and educational purposes only.

---

## Citation
If you use the ViHOS dataset or its annotation logic, please cite the original paper:

```
@inproceedings{hoang-etal-2023-vihos,
  title = {ViHOS: Vietnamese Hate and Offensive Spans Detection},
  author = {Hoang, Phu Gia and Luu, Canh Duc and Tran, Khanh Quoc and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  year = {2023},
  publisher = {Association for Computational Linguistics}
}
```

---

## Author
Student Project – Vietnamese Hate Speech Detection