import yaml  # Thư viện PyYAML để đọc file yaml
import os


class Config:
    def __init__(self, config_path="config.yaml"):
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file: {config_path}")

        # Mở file và đọc nội dung
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

    # Hàm này giúp lấy đường dẫn data an toàn
    @property
    def data(self):
        return self._cfg.get("data", {})


# Tạo sẵn một biến 'config' để các file khác chỉ việc import và dùng
# Nếu lỗi thì gán bằng None
try:
    config = Config()
except Exception:
    config = None