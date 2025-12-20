# src/services/predictor.py
import torch
from transformers import AutoTokenizer
from src.models.phobert_classifier import HateSpeechClassifier
from src.services.preprocessing.pipeline import PreprocessingPipeline


class HateSpeechPredictor:
    def __init__(self, model_path: str, device: str = 'cpu'):
        # Inference path: deterministic, no gradients; device selection affects latency and memory only
        self.device = torch.device(device)
        self.pipeline = PreprocessingPipeline()

        # Tokenizer must match PhoBERT backbone to keep vocabulary/segmentation consistent
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        # Architecture mirrors training-time model to ensure weight compatibility
        self.model = HateSpeechClassifier(n_classes=2)

        # Load weights serialized during training; eval() disables stochastic layers for stable predictions
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("--> Đã load model thành công!")
        except Exception as e:
            print(f"Lỗi load model: {e}")
            raise e

        # Fixed label mapping for binary output; change requires retraining or post-processing update
        self.idx2label = {0: "CLEAN", 1: "TOXIC"}

    def predict(self, text: str):
        # Preprocessing must mirror training-time transformations to avoid distribution shift
        clean_text = self.pipeline.process_text(text)

        encoding = self.tokenizer.encode_plus(
            clean_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Inference produces logits; softmax used only for reporting confidence, not decision thresholds
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        return {
            "text_input": text,
            "text_clean": clean_text,
            "label": self.idx2label[pred_idx],
            "confidence": f"{confidence:.2%}"
        }