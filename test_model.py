import torch
from src.models.phobert_classifier import HateSpeechClassifier


def test_model_architecture():
    print("--> Đang tải Model (Lần đầu sẽ mất 1-2 phút để tải ~500MB)...")
    model = HateSpeechClassifier(n_classes=3)

    # Tạo dữ liệu giả (Batch size = 1, độ dài câu = 10)
    fake_ids = torch.randint(0, 1000, (1, 10))  # 1 câu, 10 từ ngẫu nhiên
    fake_mask = torch.ones((1, 10))  # Mask toàn số 1

    print("--> Đang chạy thử Forward pass...")
    # Thử đút dữ liệu vào model
    output = model(fake_ids, fake_mask)

    print(f"Kích thước đầu ra: {output.shape}")
    print(f"Giá trị đầu ra (Logits): {output}")

    # Kiểm tra logic
    if output.shape == (1, 3):
        print("\n✅ THÀNH CÔNG! Model đã trả ra 3 con số (Clean, Offensive, Hate).")
    else:
        print("\n❌ THẤT BẠI! Kích thước đầu ra không đúng.")


if __name__ == "__main__":
    test_model_architecture()