import re


class TextCleaner:
    def __init__(self):
        pass

    def to_lower(self, text: str) -> str:
        return text.lower()

    def replace_special_tokens(self, text: str) -> str:
        """
        [MỚI] Thay thế emoji và dấu câu đặc biệt thành token có ý nghĩa.
        Để PhoBERT hiểu được thái độ người viết.
        """
        # 1. Xử lý Emoji Cười -> [CƯỜI]
        # Regex gom nhóm: :), :)), =)), :D, =D, 😂, 🤣
        # Cấu trúc: [:=] (mắt), -? (mũi có hoặc không), \)+ (mồm ngoặc đóng nhiều lần)
        text = re.sub(r'(:=|=)?-?\)+|😂+|🤣+|k{2,}', ' emoji_vui ', text)

        text = re.sub(r'3///|3que|3\s*que', ' phản_động ', text)
        # 2. Xử lý dấu câu nhấn mạnh
        # ??? -> [HỎI_GẮT], !!! -> [HÉT]
        # Lưu ý: Token nên viết liền (underscore) để Tokenizer không tách ra
        text = re.sub(r'\?{2,}', ' dấu_hỏi_gắt ', text)
        text = re.sub(r'!{2,}', ' dấu_chấm_than_gắt ', text)
        text = re.sub(r'\.{3,}', ' dấu_ba_chấm ', text)

        return text

    def remove_special_chars(self, text: str) -> str:
        """
        SỬA: Giữ lại dấu câu cơ bản (. ? !) vì nó ngắt câu, quan trọng cho ngữ nghĩa.
        Chỉ xóa các ký tự rác thực sự (@, #, $, %, ^, &...)
        """
        # Regex cũ: r'[^\w\s]' -> Xóa hết
        # Regex mới: Thêm .,?! vào danh sách giữ lại
        # Lưu ý: Thêm cả underscore _ để giữ lại các token mình vừa tạo ở trên (emoji_vui)
        return re.sub(r'[^\w\s\dàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,?!_]', ' ', text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_repeating_chars(self, text: str) -> str:
        """
        nguuu -> ngu
        kẹttttttt -> kẹt
        (Chỉ áp dụng khi ký tự lặp lại từ 3 lần trở lên để tránh sửa sai từ 'xoong', 'quần soóc')
        """

        return re.sub(r'(.)\1{2,}', r'\1', text)

    def run(self, text: str) -> str:
        text = self.replace_special_tokens(text)  # Chạy cái này trước để map emoji
        text = self.to_lower(text)
        text = self.normalize_repeating_chars(text)
        text = self.remove_special_chars(text)  # Sau đó mới dọn rác
        text = self.normalize_whitespace(text)
        return text