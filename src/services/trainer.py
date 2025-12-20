import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np


class HateSpeechTrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, device: str, lr: float = 2e-5):
        """
        Trainer owns optimization lifecycle for supervised classification.
        Assumes model emits logits per class and DataLoaders yield dict batches with input_ids, attention_mask, labels.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)

        # Cross-entropy aligns with multi-class logits; label IDs must be contiguous starting at 0
        self.criterion = nn.CrossEntropyLoss()

        # AdamW is standard for Transformer fine-tuning; weight decay handled internally
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def compute_metrics(self, preds, labels):
        """Return accuracy and macro-F1; macro treats classes equally, useful under imbalance."""
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return acc, f1

    def train_one_epoch(self, epoch_index):
        # Training mode enables stochastic layers; evaluation uses a separate pass
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_index}")

        for batch in progress_bar:
            # Batches must fit in device memory; failing here indicates batch size misconfiguration
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask)

            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            # Accumulate for epoch-level metrics; detach to avoid graph retention
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc, f1 = self.compute_metrics(all_preds, all_labels)

        return avg_loss, acc, f1

    def evaluate(self):
        # Inference-only path; gradients and stochastic layers must be disabled for stable metrics
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc, f1 = self.compute_metrics(all_preds, all_labels)

        return avg_loss, acc, f1

    def save_model(self, path: str):
        # Persisting state_dict enables later rehydration for inference/API without full training context
        torch.save(self.model.state_dict(), path)
        print(f"--> Đã lưu model tại: {path}")