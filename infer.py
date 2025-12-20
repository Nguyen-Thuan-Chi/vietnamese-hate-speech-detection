import torch
from src.services.predictor import HateSpeechPredictor


def main():
    # Chọn model Epoch 3 (Ngon nhất)
    MODEL_PATH = "models/phobert_epoch_3.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Đang khởi tạo Predictor trên {device.upper()}...")

    try:
        # Lưu ý: Model train với n_classes=2 thì lúc load cũng phải y hệt
        predictor = HateSpeechPredictor(MODEL_PATH, device=device)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    print("\n=== TEST MODEL TOXIC DETECTION (Binary) ===")
    print("Mời bạn nhập câu cần test (Gõ 'exit' để thoát):")

    while True:
        text = input("\n>> Nhập: ")
        if text.lower() in ['exit', 'quit']:
            break
        if not text.strip(): continue

        result = predictor.predict(text)

        # In màu mè tí cho dễ nhìn
        label = result['label']
        conf = result['confidence']

        print("-" * 50)
        print(f"Gốc:   {result['text_input']}")
        print(f"Sạch:  {result['text_clean']}")

        if label == "TOXIC":
            print(f"Kết quả: 🔴 {label} (Độ tin cậy: {conf})")
        else:
            print(f"Kết quả: 🟢 {label} (Độ tin cậy: {conf})")
        print("-" * 50)


if __name__ == "__main__":
    main()