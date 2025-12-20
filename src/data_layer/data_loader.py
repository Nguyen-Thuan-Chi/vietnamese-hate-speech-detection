# src/data_layer/data_loader.py
import pandas as pd
from typing import List
from src.core.dtos import HateSpeechSample
from src.core.interfaces import IDataLoader


class DataLoader(IDataLoader):
    """
    Loader for ViHOS datasets. Supports sequence-labeled inputs by aggregating tokens per sentence_id.
    Assumes UTF-8 CSV with columns consistent to the dataset version; falls back to sentence-level fields when needed.
    """

    def load_data(self, file_path: str) -> List[HateSpeechSample]:
        print(f"--> [DataLoader] Đang đọc file từ: {file_path}")
        results = []

        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            # Sequence format detected: reconstruct text and preserve tag lists for downstream conversion
            if 'sentence_id' in df.columns and 'Word' in df.columns:
                print("--> Phát hiện dữ liệu dạng Sequence (Từ tách rời). Đang ghép lại thành câu...")

                def join_text(x):
                    return " ".join([str(s) for s in x])

                grouped_df = df.groupby('sentence_id').agg({
                    'Word': join_text,
                    'Tag': list
                }).reset_index()

                for _, row in grouped_df.iterrows():
                    full_text = row['Word']
                    tags_str = str(row['Tag'])
                    sample = HateSpeechSample(text=full_text, label=tags_str)
                    results.append(sample)

            else:
                # Sentence-level fallback; column names vary across sources
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
            traceback.print_exc()
            return []