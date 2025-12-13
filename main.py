# main.py
from src.utils.config_loader import config
from src.data_layer.data_loader import DataLoader


def main():
    # ... (code bên trong giữ nguyên) ...
    print("=== TEST LOAD DỮ LIỆU ===")

    if config is None:
        print("Lỗi: Không tìm thấy file config.yaml")
        return

    path = config.data.get('train_path')
    print(f"File cần đọc: {path}")

    loader = DataLoader()
    data = loader.load_data(path)

    if data:
        print(f"\nThành công! Tìm thấy {len(data)} mẫu dữ liệu.")
        print("Ví dụ 3 dòng đầu tiên:")
        for item in data[:3]:
            print(f" - {item}")
    else:
        print("\nThất bại. Danh sách rỗng.")


# --- ĐÂY LÀ PHẦN BẠN ĐANG THIẾU ---
# Nó phải nằm sát lề trái (không được thụt vào trong)
if __name__ == "__main__":
    main()