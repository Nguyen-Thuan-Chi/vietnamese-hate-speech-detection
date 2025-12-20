# check_typo.py
from src.services.preprocessing.pipeline import PreprocessingPipeline


def check_weird_cases():
    pipeline = PreprocessingPipeline()

    # Những ca "khó đỡ" mà bạn lo lắng
    test_cases = [
        "thawngf nayf laf ai",  # Lỗi bộ gõ Telex
        "ddungf co roi",  # Lỗi bộ gõ d/đ
        "nguuu quáaa điiii",  # Kéo dài ký tự (Dup keys)
        "dkm dcm dm",  # Teencode chửi thề (Check lại xem map đúng chưa)
        "bàn phím kẹt kẹttttttt",  # Kẹt phím
        "Dừa lắm :)))) =)))",  # Emoji + Teencode
        "cái loz gì thế"  # Từ khóa Hate Speech quan trọng
    ]

    print(f"{'INPUT (GỐC)':<30} | {'OUTPUT (SAU KHI XỬ LÝ)':<30}")
    print("-" * 70)

    for text in test_cases:
        clean_text = pipeline.process_text(text)
        print(f"{text:<30} | {clean_text}")


if __name__ == "__main__":
    check_weird_cases()