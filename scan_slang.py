import pandas as pd
from collections import Counter
import re

# Cấu hình đường dẫn file data của bạn
FILE_PATH = "data/Sequence_labeling_based_version/Syllable/train_BIO_syllable.csv"


def is_teencode_suspect(word):
    # Những từ chứa ký tự đặc trưng của teencode: j, w, f, z
    # Hoặc lặp ký tự: "nguu", "đcccc"
    if re.search(r'[jwfz]', word):
        return True
    if re.search(r'(.)\1{2,}', word):  # Ký tự lặp lại 3 lần trở lên (vd: lozzz)
        return True
    return False


def main():
    print(f"--> Đang quét file: {FILE_PATH} ...")

    try:
        # Đọc cột 'Word' từ file CSV
        df = pd.read_csv(FILE_PATH, encoding='utf-8')

        # Lấy tất cả các từ, chuyển về chữ thường, bỏ giá trị rỗng
        all_words = df['Word'].dropna().astype(str).str.lower().tolist()

        # Đếm tần suất
        word_counts = Counter(all_words)

        print(f"\n=== TỔNG SỐ TỪ VỰNG: {len(word_counts)} ===")

        print("\n--- TOP 30 TỪ CHỨA 'J, W, F, Z' (Teencode chúa) ---")
        suspects = {w: c for w, c in word_counts.items() if is_teencode_suspect(w)}
        # Sắp xếp theo độ phổ biến giảm dần
        sorted_suspects = sorted(suspects.items(), key=lambda x: x[1], reverse=True)[:30]
        for w, c in sorted_suspects:
            print(f"{w}: {c} lần")

        print("\n--- TOP 50 TỪ XUẤT HIỆN NHIỀU NHẤT (Kiểm tra xem có sót từ nào không) ---")
        # In ra top 50 từ phổ biến nhất
        for w, c in word_counts.most_common(50):
            print(f"{w}: {c}")

    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()