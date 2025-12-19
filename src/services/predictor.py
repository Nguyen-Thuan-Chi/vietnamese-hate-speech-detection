# src/services/predictor.py
import torch
from transformers import AutoTokenizer
from src.models.phobert_classifier import HateSpeechClassifier
from src.services.preprocessing.pipeline import PreprocessingPipeline


class HateSpeechPredictor:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.pipeline = PreprocessingPipeline()

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        # 2. Load Model Architecture
        self.model = HateSpeechClassifier(n_classes=2)

        # 3. Load Model Weights (Cái file .pth 500MB)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Quan trọng: Tắt chế độ training (Dropout)
            print("--> Đã load model thành công!")
        except Exception as e:
            print(f"Lỗi load model: {e}")
            raise e

        # Label Mapping
        self.idx2label = {0: "CLEAN", 1: "TOXIC"}

    def predict(self, text: str):
        # 1. Preprocess (Cleaning + Teencode)
        clean_text = self.pipeline.process_text(text)

        # 2. Tokenize
        encoding = self.tokenizer.encode_plus(
            clean_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            # outputs là logits [0.1, -0.5, 2.3]

            # Dùng Softmax để tính % tự tin
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Lấy index có điểm cao nhất
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        return {
            "text_input": text,
            "text_clean": clean_text,
            "label": self.idx2label[pred_idx],
            "confidence": f"{confidence:.2%}"
        }