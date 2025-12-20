from .cleaning import TextCleaner
from .teencode import TeencodeConverter
from typing import List
from src.core.dtos import HateSpeechSample


class PreprocessingPipeline:
    def __init__(self):
        # Thuê 2 nhân viên về làm việc
        self.cleaner = TextCleaner()
        self.teencode_converter = TeencodeConverter()

    def process_text(self, text: str) -> str:
        """Xử lý 1 câu văn bản"""
        # 1. Dọn rác trước
        text = self.cleaner.run(text)
        # 2. Dịch teencode sau (vì teencode cần text sạch để tách từ chính xác)
        text = self.teencode_converter.convert(text)
        return text

    def run(self, data: List[HateSpeechSample]) -> List[HateSpeechSample]:
        print("--> [Pipeline] Đang làm sạch dữ liệu...")
        processed_data = []

        for item in data:
            clean_text = self.process_text(item.text)
            label_str = item.label

            # --- LOGIC NHỊ PHÂN (0 vs 1) ---
            # Chỉ có tag B-T và I-T là độc hại
            if "B-T" in label_str or "I-T" in label_str:
                final_label = 1  # TOXIC
            else:
                final_label = 0  # CLEAN

            processed_data.append(HateSpeechSample(text=clean_text, label=str(final_label)))

        print(f"--> [Pipeline] Xong! Đã xử lý {len(processed_data)} dòng.")
        return processed_data