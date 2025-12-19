import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm  # Thanh hiển thị tiến độ
import numpy as np


class HateSpeechTrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, device: str, lr: float = 2e-5):
        """
        model: Model PhoBERT đã tạo
        train_loader: Dữ liệu train (chia batch)
        val_loader: Dữ liệu kiểm thử (chia batch)
        device: 'cuda' (GPU) hoặc 'cpu'
        lr: Learning rate (tốc độ học), mặc định 2e-5 (chuẩn cho BERT)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)  # Đẩy model vào GPU/CPU

        # 1. Hàm mất mát (Loss Function)
        # CrossEntropyLoss là chuẩn bài cho bài toán phân loại nhiều lớp (0, 1, 2)
        self.criterion = nn.CrossEntropyLoss()

        # 2. Thuật toán tối ưu (Optimizer)
        # AdamW là biến thể của Adam, fix lỗi weight decay, chuyên dùng cho Transformer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def compute_metrics(self, preds, labels):
        """Tính Accuracy và F1-Macro"""
        preds = np.argmax(preds, axis=1)  # Chuyển logits thành class (ví dụ [0.1, 0.9, 0] -> class 1)
        acc = accuracy_score(labels, preds)
        # F1 Macro: Tính trung bình F1 của từng lớp, quan trọng khi dữ liệu mất cân bằng
        f1 = f1_score(labels, preds, average='macro')
        return acc, f1

    def train_one_epoch(self, epoch_index):
        self.model.train()  # Bật chế độ train (để Dropout hoạt động)
        total_loss = 0
        all_preds = []
        all_labels = []

        # Tqdm tạo thanh loading bar
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_index}")

        for batch in progress_bar:
            # 1. Đẩy dữ liệu vào device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 2. Xóa gradient cũ
            self.optimizer.zero_grad()

            # 3. Forward (Model đoán)
            outputs = self.model(input_ids, attention_mask)

            # 4. Tính lỗi (Loss)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            # 5. Backward (Tìm lỗi)
            loss.backward()

            # 6. Update (Sửa lỗi)
            self.optimizer.step()

            # Lưu kết quả để tính metrics
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            # Cập nhật loss lên thanh loading
            progress_bar.set_postfix({'loss': loss.item()})

        # Tính toán metric cuối epoch
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc, f1 = self.compute_metrics(all_preds, all_labels)

        return avg_loss, acc, f1

    def evaluate(self):
        self.model.eval()  # Bật chế độ đánh giá (tắt Dropout)
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Tắt tính đạo hàm cho nhẹ máy
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
        """Lưu model để sau này dùng làm API"""
        torch.save(self.model.state_dict(), path)
        print(f"--> Đã lưu model tại: {path}")