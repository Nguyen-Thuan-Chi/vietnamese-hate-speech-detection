# debug_data.py
from src.data_layer.data_loader import DataLoader
from src.services.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config_loader import config
from collections import Counter


def check_label_distribution():
    print("=== KIỂM TRA PHÂN BỐ NHÃN DỮ LIỆU ===")

    # 1. Load dữ liệu thô
    path = config.data.get('train_path')
    print(f"--> Đọc file: {path}")
    loader = DataLoader()
    raw_data = loader.load_data(path)

    # In thử nhãn gốc của 5 dòng đầu tiên
    print("\n--- 5 DÒNG ĐẦU TIÊN (RAW) ---")
    for i in range(5):
        print(f"Label gốc: {raw_data[i].label} | Text: {raw_data[i].text[:50]}...")

    # 2. Chạy qua Pipeline (chỗ nghi vấn bị lỗi)
    pipeline = PreprocessingPipeline()
    clean_data = pipeline.run(raw_data)

    # 3. Đếm số lượng nhãn sau khi xử lý
    labels = [d.label for d in clean_data]
    count = Counter(labels)

    print("\n--- KẾT QUẢ PHÂN BỐ SAU KHI PIPELINE XỬ LÝ ---")
    print(f"Total: {len(labels)}")
    print(f"Class 0 (Clean)    : {count.get('0', 0)}")
    print(f"Class 1 (Offensive): {count.get('1', 0)}")
    print(f"Class 2 (Hate)     : {count.get('2', 0)}")

    if count.get('1', 0) == 0 and count.get('2', 0) == 0:
        print("\n❌ LỖI NGHIÊM TRỌNG: Toàn bộ dữ liệu đã bị biến thành Clean (0).")
        print("Model học toàn số 0 nên F1 = 1.0 là đúng rồi!")
        print("--> Cần sửa logic trong src/services/preprocessing/pipeline.py")
    else:
        print("\n✅ Dữ liệu có vẻ ổn về mặt phân bố. Vấn đề nằm ở chỗ khác.")


if __name__ == "__main__":
    check_label_distribution()