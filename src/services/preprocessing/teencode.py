# src/services/preprocessing/teencode.py
import re

class TeencodeConverter:
    def __init__(self):
        self.teencode_dict = {
            # --- Nhóm phủ định (Quan trọng nhất) ---
            "k": "không", "ko": "không", "kh": "không", "k0": "không", "hok": "không",
            "hông": "không", "khum": "không", "hem": "không",

            # --- Nhóm đại từ xưng hô ---
            "ck": "chồng", "vk": "vợ", "vc": "vợ chồng",
            "t": "tao", "m": "mày", "mik": "mình", "mk": "mình",
            "ng": "người", "mn": "mọi người",
            "ad": "ad",  # (Ok, thích thì đổi cũng được, nhưng không bắt buộc)

            # --- Nhóm từ nối/thông dụng ---
            "dc": "được", "đc": "được", "dk": "được",
            "j": "gì", "trc": "trước", "ntn": "như thế nào",
            "ncl": "nói chung là", "vs": "với", "wa": "quá", "wá": "quá",
            "bh": "bây giờ",
            "lun": "luôn", "lm": "làm",
            "fb": "facebook", "ib": "inbox",

            # --- Nhóm CHỬI THỀ / CẢM THÁN (Hate Speech Key) ---
            # Model cần hiểu rõ mấy từ này để bắt Hate Speech
            "vl": "tục_tĩu",
            "cl": "tục_tĩu",
            "dcm": "địt con mẹ", "đcm": "địt con mẹ", "đkm": "địt con mẹ",
            "dm": "địt mẹ", "đm": "địt mẹ",
            "vcc": "tục_tĩu",
            "vcl": "tục_tĩu",
            "cc": "tục_tĩu",  # Hoặc con c**, tùy ngữ cảnh
            "ml": "mặt l**",
            # Explicit insults with clear targets are expanded.
            # Pure intensifiers or ambiguous vulgar slang are mapped to "tục_tĩu".

            "oc": "óc chó", "occ": "óc chó",
            # 1. Xử lý chính trị (Bắt buộc map vì bọn này hay lách luật viết tắt)

        }

    def convert(self, text: str) -> str:
        # Regex này an toàn hơn \b với tiếng Việt:
        # (?<!\w) nghĩa là phía trước không phải chữ
        # (?!\w) nghĩa là phía sau không phải chữ
        # Giúp thay thế chính xác từ đơn lẻ "k" mà không thay thế chữ "k" trong từ "kênh"
        pattern = r'(?<!\w)(' + '|'.join(re.escape(key) for key in self.teencode_dict.keys()) + r')(?!\w)'

        # Hàm callback để thay thế
        return re.sub(pattern, lambda x: self.teencode_dict[x.group()], text)
