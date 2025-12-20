# scan_tags.py
import pandas as pd
from collections import Counter
from src.utils.config_loader import config


def main():
    path = config.data.get("train_path")
    if not path:
        print("❌ Không tìm thấy train_path trong config")
        return

    print(f"Reading {path}...")
    df = pd.read_csv(path)

    # Kiểm tra cột cần thiết
    required_cols = {"Word", "Tag", "sentence_id"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Thiếu cột: {missing}")
        print(f"Các cột hiện có: {df.columns.tolist()}")
        return

    # Lấy toàn bộ tag token-level
    tags = df["Tag"].astype(str)

    tag_counter = Counter(tags)

    print("\n=== DANH SÁCH TAG (TOKEN-LEVEL) ===")
    for tag, count in tag_counter.most_common():
        print(f"Tag: {tag:>5} | Count: {count}")

    print("\nTổng số tag khác nhau:", len(tag_counter))


if __name__ == "__main__":
    main()
