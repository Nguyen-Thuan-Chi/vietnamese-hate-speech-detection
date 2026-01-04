# Vietnamese Hate Speech Detection System — Technical Reference

> **Full documentation for developers and researchers.**  
> For a quick overview, see [README.md](README.md).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Model Description](#model-description)
4. [Highlighting Strategy](#highlighting-strategy)
5. [Explainability (LIME)](#explainability-lime)
6. [Feedback Loop](#feedback-loop)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [API Reference](#api-reference)
10. [Project Structure](#project-structure)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [Citation](#citation)

---

## Project Overview

This repository contains an academic application for detecting Vietnamese hate and offensive text. The system performs **sentence-level binary classification** (TOXIC vs. CLEAN) using a fine-tuned PhoBERT model.

| Attribute | Value |
|-----------|-------|
| Language | Vietnamese |
| Task | Binary classification (sentence-level) |
| Model | PhoBERT (`vinai/phobert-base-v2`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit dashboard |
| Dataset | ViHOS (adapted) |

> **Important:** This project does **not** perform token-level or span-level prediction. The original ViHOS dataset contains word-level BIO annotations, but these are collapsed into binary sentence-level labels during training.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                   (Streamlit Dashboard)                      │
│         - Single text analysis with LIME explanation         │
│         - Batch CSV/XLSX processing                          │
│         - User feedback collection                           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      Backend API                             │
│                  (FastAPI + Uvicorn)                         │
│         - /predict endpoint for inference                    │
│         - Input validation and error handling                │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                      │
│         - Text cleaning (normalization, regex)               │
│         - Teencode conversion (Vietnamese slang)             │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    PhoBERT Classifier                        │
│         - vinai/phobert-base-v2 backbone                     │
│         - Dropout (0.3) + Linear classification head         │
│         - Output: CLEAN (0) or TOXIC (1) + confidence        │
└──────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| `src/dashboard/app.py` | Streamlit UI for interactive analysis |
| `src/api/server.py` | FastAPI backend serving predictions |
| `src/services/predictor.py` | Model inference logic |
| `src/services/preprocessing/` | Text cleaning and normalization |
| `src/services/highlighter.py` | Keyword-based highlighting |
| `src/services/explainer.py` | LIME-based explanation service |
| `src/services/feedback.py` | User feedback persistence |
| `src/models/phobert_classifier.py` | Model architecture definition |

---

## Model Description

### Architecture

The classifier uses **PhoBERT** (`vinai/phobert-base-v2`) as the backbone with a classification head:

```
PhoBERT Encoder → Pooled Output → Dropout(0.3) → Linear(768, 2)
```

- **Input:** Tokenized Vietnamese text (max 128 tokens)
- **Output:** Logits for 2 classes (CLEAN, TOXIC)
- **Inference:** Softmax applied for confidence scores

### Training Data Transformation

The original ViHOS dataset contains word-level BIO annotations (`B-T`, `I-T` for toxic spans, `O` for non-toxic). This project **collapses** these annotations into sentence-level labels:

- If a sentence contains **any** `B-T` or `I-T` tag → **TOXIC (1)**
- Otherwise → **CLEAN (0)**

### Why Span-Level Prediction Is Not Available

The current model architecture uses the **pooled sentence representation** for classification. It does not include a token classification head. Therefore:

- The model outputs **only sentence-level predictions**
- There is **no span extraction** at inference time
- The `spans` field in API responses is always an **empty list**

Implementing span-level prediction would require:
1. A token classification architecture (e.g., BIO tagging head)
2. Retraining on the original word-level annotations
3. Post-processing to reconstruct character spans from tokens

---

## Highlighting Strategy

When the model predicts a sentence as **TOXIC**, the UI indicates potentially problematic words using a **keyword-based fallback** mechanism.

### How It Works

1. A curated list of Vietnamese toxic keywords is maintained in `src/config/toxic_keywords.py`
2. When a TOXIC prediction is made, the `KeywordHighlighter` scans the input text
3. Matching keywords are highlighted in the UI
4. A disclaimer is displayed to users

### Disclaimer

> ⚠️ **The highlighted keywords are NOT model predictions.** They are heuristic matches from a predefined vocabulary. The model determines whether the entire sentence is toxic; it does not identify specific offensive words or spans.

---

## Explainability (LIME)

For TOXIC predictions with sufficient text length (≥5 words), the system provides **LIME explanations**.

### How It Works

- LIME perturbs the input text and observes prediction changes
- Word-level importance scores are computed
- Results are visualized as a bar chart in the UI

### Important Note

LIME explanations are **post-hoc interpretations**. They approximate which words influenced the model's decision but are not direct model outputs. LIME treats the model as a black box.

---

## Feedback Loop

### Collection Mechanism

The Streamlit dashboard includes feedback buttons:

1. Users can mark predictions as **correct** or **incorrect**
2. For incorrect predictions, users can provide a description
3. Feedback is appended to `data/feedback.csv`

### Data Format

```csv
text,predicted_spans,user_feedback
"example text","[]","correct"
"another example","[]","incorrect: missed toxic content..."
```

### Intended Use

- **Analysis:** Understanding model failure modes
- **Dataset curation:** Incorporating corrections into future training data
- **Evaluation:** Measuring real-world performance

> **Note:** Automatic retraining based on feedback is **not implemented**. Feedback is stored for manual review only.

---

## Installation & Setup

### Prerequisites

- **Python:** 3.8+
- **OS:** Linux / macOS
- **Hardware:** CPU sufficient; GPU optional

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ViHOS
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Package (Optional)

```bash
pip install -e .
```

### Step 5: Download Model Weights

Model checkpoints are **not included** in this repository. Place the checkpoint file in:

```
models/phobert_epoch_3.pth
```

> The application will not start without this file.

### Step 6: Verify Configuration

Check `config.yaml`:

```yaml
data:
  train_path: "data/Sequence_labeling_based_version/Syllable/train_BIO_syllable.csv"

system:
  device: "cpu"
```

---

## Running the Application

### Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

Access: **http://localhost:8501**

### FastAPI Backend

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Access: **http://localhost:8000/docs**

---

## API Reference

### GET /

Health check.

```json
{"status": "healthy", "device": "cpu"}
```

### POST /predict

**Request:**
```json
{"text": "Nội dung cần kiểm tra"}
```

**Response:**
```json
{
  "label": "TOXIC",
  "confidence": "99.69%",
  "clean_text": "nội dung cần kiểm tra"
}
```

**Constraints:** Text must not be empty; maximum 2000 characters.

---

## Project Structure

```
ViHOS/
├── src/
│   ├── api/              # FastAPI server
│   ├── config/           # Toxic keywords list
│   ├── core/             # Data structures
│   ├── dashboard/        # Streamlit UI
│   ├── data_layer/       # Data loading
│   ├── models/           # Model architecture
│   ├── services/         # Business logic
│   │   ├── preprocessing/
│   │   ├── predictor.py
│   │   ├── highlighter.py
│   │   ├── explainer.py
│   │   └── feedback.py
│   └── utils/            # Configuration, metrics
├── data/                 # Training data (ViHOS)
├── models/               # Model checkpoints (not in repo)
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## Limitations

| Limitation | Description |
|------------|-------------|
| Sentence-level only | Classifies entire sentences, not individual words |
| No span prediction | Does not extract toxic spans despite ViHOS annotations |
| Keyword highlighting is heuristic | Highlighted words are from a curated list, not model outputs |
| Vietnamese only | Trained exclusively on Vietnamese text |
| No real-time learning | Feedback is stored but not used for automatic updates |

---

## Future Work

1. Token classification model for span-level prediction
2. Feedback integration pipeline for retraining
3. Multi-class classification (hate speech categories)
4. Model versioning and comparison

---

## Citation

If you use the ViHOS dataset, please cite:

```bibtex
@inproceedings{hoang-etal-2023-vihos,
  title = {ViHOS: Vietnamese Hate and Offensive Spans Detection},
  author = {Hoang, Phu Gia and Luu, Canh Duc and Tran, Khanh Quoc and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  year = {2023},
  publisher = {Association for Computational Linguistics}
}
```

---

## License

See [LICENSE](LICENSE) for details.

