import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.utils.config_loader import config
from src.data_layer.data_loader import DataLoader as MyDataLoader
from src.services.preprocessing.pipeline import PreprocessingPipeline
from src.core.dataset import HateSpeechDataset
from src.models.phobert_classifier import HateSpeechClassifier
from src.services.trainer import HateSpeechTrainer


def main():
    print("=== HỆ THỐNG HUẤN LUYỆN HATE SPEECH DETECTION (3 CLASSES) ===")

    # 1. Config & Device
    if config is None: return
    # Tự động chọn GPU nếu có, không thì CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Đang chạy trên thiết bị: {device.upper()}")

    # 2. Load & Preprocess Data
    raw_path = config.data.get('train_path')
    loader = MyDataLoader()
    raw_data = loader.load_data(raw_path)

    pipeline = PreprocessingPipeline()
    clean_data = pipeline.run(raw_data)  # Data đã có label 0, 1, 2

    # 3. Chia tập Train (80%) và Validation (20%)
    # Stratify giúp đảm bảo tỉ lệ Hate/Clean ở 2 tập là như nhau (quan trọng!)
    labels = [int(d.label) for d in clean_data]
    train_data, val_data = train_test_split(
        clean_data,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"--> Dữ liệu: Train ({len(train_data)}) | Val ({len(val_data)})")

    # 4. Tokenizer & Dataset
    print("--> Đang tải Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    train_dataset = HateSpeechDataset(train_data, tokenizer)
    val_dataset = HateSpeechDataset(val_data, tokenizer)

    # 5. DataLoaders (Xe chở hàng)
    # Batch size: Số lượng câu học cùng lúc.
    # Nếu máy yếu (CPU) để 8 hoặc 16. Nếu có GPU xịn để 32.
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 6. Khởi tạo Model & Trainer
    print("--> Đang khởi tạo Model...")
    model = HateSpeechClassifier(n_classes=2)
    trainer = HateSpeechTrainer(model, train_loader, val_loader, device=device)

    # 7. Training Loop (Chạy thử 3 Epochs xem sao)
    EPOCHS = 3
    print(f"\n--> BẮT ĐẦU TRAIN ({EPOCHS} epochs)...")

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc, train_f1 = trainer.train_one_epoch(epoch)

        # Validate
        val_loss, val_acc, val_f1 = trainer.evaluate()

        print(f"\n--- EPOCH {epoch} KẾT QUẢ ---")
        print(f"Train Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | F1: {val_f1:.4f}")
        print("-" * 50)

        # Lưu model sau mỗi epoch (Save Checkpoint)
        trainer.save_model(f"models/phobert_epoch_{epoch}.pth")

    print("\n--> HOÀN TẤT HUẤN LUYỆN!")


if __name__ == "__main__":
    main()