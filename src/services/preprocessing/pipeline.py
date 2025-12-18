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
        """Xử lý cả 1 danh sách (Batch processing)"""
        print("--> [Pipeline] Đang làm sạch dữ liệu...")
        processed_data = []

        for item in data:
            # Xử lý text
            clean_text = self.process_text(item.text)

            # Xử lý Label (Chuyển từ list string "['O', 'B-H']" sang số "1")
            # Logic: Nếu có nhãn 'B-' hoặc 'I-' (tức là có nhãn xấu) thì gán là 1, ngược lại là 0
            label_str = item.label
            is_hate = 1 if ("B-" in label_str or "I-" in label_str) else 0

            # Đóng gói lại vào hộp mới
            processed_data.append(HateSpeechSample(text=clean_text, label=str(is_hate)))

        print(f"--> [Pipeline] Xong! Đã xử lý {len(processed_data)} dòng.")
        return processed_data