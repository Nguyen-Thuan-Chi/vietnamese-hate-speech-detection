import yaml
import os


class Config:
    def __init__(self, config_path="config.yaml"):
        # Fail early if configuration is missing; prevents partial runs with undefined paths
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

    @property
    def data(self):
        # Data section holds dataset paths and related settings; defaults to empty for robustness
        return self._cfg.get("data", {})


# Provide a module-level config for convenience; downstream code should handle None defensively
try:
    config = Config()
except Exception:
    config = None