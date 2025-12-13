# src/data_layer/data_loader.py
import pandas as pd
from typing import List
from src.core.dtos import HateSpeechSample
from src.core.interfaces import IDataLoader


class DataLoader(IDataLoader):
    """
    Class chịu trách nhiệm đọc dữ liệu ViHOS.
    Phiên bản: Hỗ trợ file Sequence Labeling (Gom nhóm theo sentence_id)
    """

    def load_data(self, file_path: str) -> List[HateSpeechSample]:
        print(f"--> [DataLoader] Đang đọc file từ: {file_path}")
        results = []

        try:
            # 1. Đọc file CSV
            df = pd.read_csv(file_path, encoding='utf-8')

            # 2. Kiểm tra xem có cột 'sentence_id' và 'Word' không (đặc trưng của ViHOS dạng sequence)
            if 'sentence_id' in df.columns and 'Word' in df.columns:
                print("--> Phát hiện dữ liệu dạng Sequence (Từ tách rời). Đang ghép lại thành câu...")

                # Hàm để xử lý khi gom nhóm: Nối các từ lại bằng dấu cách
                def join_text(x):
                    return " ".join([str(s) for s in x])

                # Gom nhóm theo sentence_id
                # 'Word' sẽ được nối lại, 'Tag' sẽ giữ nguyên danh sách
                grouped_df = df.groupby('sentence_id').agg({
                    'Word': join_text,
                    'Tag': list  # Giữ nguyên list các tag để sau này dùng nếu cần
                }).reset_index()

                # Duyệt qua từng câu đã ghép
                for _, row in grouped_df.iterrows():
                    full_text = row['Word']

                    # Tạm thời lấy tag đầu tiên hoặc xử lý logic tag sau
                    # Ở đây mình convert list tag thành chuỗi để in ra cho dễ nhìn
                    tags_str = str(row['Tag'])

                    sample = HateSpeechSample(text=full_text, label=tags_str)
                    results.append(sample)

            else:
                # Fallback: Nếu không phải dạng sequence thì đọc kiểu cũ (dòng đơn)
                print("--> Dữ liệu dạng thường (Sentence level).")
                for _, row in df.iterrows():
                    text_val = str(row.get('sentence', row.get('text', '')))
                    label_val = str(row.get('label', row.get('tag', '')))
                    if text_val.strip():
                        results.append(HateSpeechSample(text=text_val, label=label_val))

            print(f"--> [DataLoader] Đã load xong {len(results)} câu hoàn chỉnh.")
            return results

        except Exception as e:
            print(f"[Lỗi] Không đọc được file: {e}")
            import traceback
            traceback.print_exc()  # In chi tiết lỗi để debug
            return []