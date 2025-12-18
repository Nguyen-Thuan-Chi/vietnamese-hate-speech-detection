from src.utils.config_loader import config
from src.data_layer.data_loader import DataLoader
# Import Class quản lý mới
from src.services.preprocessing.pipeline import PreprocessingPipeline


def main():
    print("=== TEST PREPROCESSING PIPELINE ===")

    # 1. Load Data (Như cũ)
    if config is None: return
    path = config.data.get('train_path')
    loader = DataLoader()
    raw_data = loader.load_data(path)

    if not raw_data: return

    # 2. Gọi Pipeline làm sạch (MỚI)
    pipeline = PreprocessingPipeline()
    clean_data = pipeline.run(raw_data)

    # 3. So sánh kết quả
    print("\n--- SO SÁNH TRƯỚC VÀ SAU KHI XỬ LÝ ---")
    # In thử 5 dòng đầu để thấy sự khác biệt
    for i in range(5):
        print(f"[Gốc ]: {raw_data[i].text} | Label: {raw_data[i].label}")
        print(f"[Sạch]: {clean_data[i].text} | Label: {clean_data[i].label}")
        print("-" * 50)


if __name__ == "__main__":
    main()