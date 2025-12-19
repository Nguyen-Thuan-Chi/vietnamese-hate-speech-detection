# src/services/preprocessing/teencode.py
import re


class TeencodeConverter:
    def __init__(self):
        self.teencode_dict = {
            # --- Nhóm phủ định ---
            "k": "không", "ko": "không", "kh": "không", "k0": "không", "hok": "không",
            "hông": "không", "khum": "không", "hem": "không",

            # --- Nhóm đại từ xưng hô ---
            "ck": "chồng", "vk": "vợ", "vc": "vợ chồng",
            "t": "tao",  "mik": "mình", "mk": "mình",
            "mn": "mọi người", "ad": "admin",

            # --- Nhóm từ nối/thông dụng ---
            "dc": "được", "đc": "được", "dk": "được",
            "j": "gì", "trc": "trước", "ntn": "như thế nào",
            "ncl": "nói chung là", "vs": "với", "wa": "quá", "wá": "quá",
            "bh": "bây giờ", "lun": "luôn", "lm": "làm",
            "fb": "facebook", "ib": "inbox",

            # --- Nhóm HATE SPEECH / SLANG ---
            # TƯ DUY MỚI: Chỉ chuẩn hóa về từ gốc (Normalization), KHÔNG gán nhãn đạo đức (Rule-based).
            # Để PhoBERT tự học ngữ cảnh (vui vl vs ngu vl).

            # 1. Nhóm chửi thề (Gom biến thể về 1 từ chuẩn để model học tập trung hơn)
            "dcm": "đcm", "đkm": "đcm", "dkm": "đcm", "đmm": "đm",  # Gom hết về đcm
            "dm": "đm", "dmm": "đm",  # Gom hết về đm
            "vcl": "vcl", "vcc": "vcc", "vch": "vch",  # Giữ nguyên sắc thái
            "cl": "cl", "ml": "ml", "cc": "cc",

            # 2. Nhóm từ viết tắt cảm thán
            "vl": "vl",  # Giữ nguyên vì có thể dùng cho câu khen (ngon vl)

            # 3. Từ xúc phạm cụ thể (Gom biến thể)
            "oc": "óc_chó", "occ": "óc_chó", "occho": "óc_chó",
            "đĩ": "đĩ",


        }

    def convert(self, text: str) -> str:
        # [QUAN TRỌNG] Sort từ dài xuống ngắn để tránh thay thế nhầm
        # Ví dụ: Xử lý "vcl" trước "vc", "nguu" trước "ngu"
        keys = sorted(self.teencode_dict.keys(), key=len, reverse=True)

        # Regex update:
        # map(re.escape, keys): Đảm bảo các ký tự đặc biệt trong key (như 3///) không gây lỗi regex
        pattern = r'(?<!\w)(' + '|'.join(map(re.escape, keys)) + r')(?!\w)'

        return re.sub(pattern, lambda x: self.teencode_dict[x.group()], text)