from dataclasses import dataclass

#dont touch it, myself will thank you later
# @dataclass là một cách viết tắt của Python
# Nó tự động tạo ra hàm __init__ để gán dữ liệu, giúp code ngắn gọn
@dataclass
class HateSpeechSample:
    text: str           # Quy định: text bắt buộc là chuỗi (string)
    label: str = None   # Quy định: label là chuỗi, mặc định là None (nếu file không có nhãn)