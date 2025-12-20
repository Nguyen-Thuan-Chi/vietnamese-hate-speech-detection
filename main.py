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

    # Config must be loaded before proceeding; silently abort if missing to avoid partial runs
    if config is None: return
    # Prefer GPU for transformer training; falls back to CPU when unavailable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Đang chạy trên thiết bị: {device.upper()}")

    # Data is expected to be labeled; pipeline will collapse sequence labels into binary classes
    raw_path = config.data.get('train_path')
    loader = MyDataLoader()
    raw_data = loader.load_data(raw_path)

    pipeline = PreprocessingPipeline()
    clean_data = pipeline.run(raw_data)

    # Preserve class distribution across splits to keep evaluation stable on imbalanced data
    labels = [int(d.label) for d in clean_data]
    train_data, val_data = train_test_split(
        clean_data,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"--> Dữ liệu: Train ({len(train_data)}) | Val ({len(val_data)})")

    # Tokenizer tied to model family; must match PhoBERT checkpoints used by the classifier
    print("--> Đang tải Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    train_dataset = HateSpeechDataset(train_data, tokenizer)
    val_dataset = HateSpeechDataset(val_data, tokenizer)

    # Batch size trades memory for throughput; adjust externally based on hardware constraints
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model and trainer are instantiated per run; checkpoints captured every epoch for reproducibility
    print("--> Đang khởi tạo Model...")
    model = HateSpeechClassifier(n_classes=2)
    trainer = HateSpeechTrainer(model, train_loader, val_loader, device=device)

    # Keep a short training horizon in the script; longer runs should be configured via experiment tooling
    EPOCHS = 3
    print(f"\n--> BẮT ĐẦU TRAIN ({EPOCHS} epochs)...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = trainer.train_one_epoch(epoch)
        val_loss, val_acc, val_f1 = trainer.evaluate()

        print(f"\n--- EPOCH {epoch} KẾT QUẢ ---")
        print(f"Train Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | F1: {val_f1:.4f}")
        print("-" * 50)

        # Persist epoch-level checkpoints to enable later selection based on validation metrics
        trainer.save_model(f"models/phobert_epoch_{epoch}.pth")

    print("\n--> HOÀN TẤT HUẤN LUYỆN!")


if __name__ == "__main__":
    main()