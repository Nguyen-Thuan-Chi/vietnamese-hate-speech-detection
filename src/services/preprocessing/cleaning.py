import re


class TextCleaner:
    def __init__(self):
        pass

    def to_lower(self, text: str) -> str:
        return text.lower()

    def replace_special_tokens(self, text: str) -> str:
        """
        Map emojis and emphatic punctuation to stable tokens to preserve sentiment cues under tokenization.
        """
        # Consolidate common laughter/emoji patterns into a single semantic token
        text = re.sub(r'(:=|=)?-?\)+|😂+|🤣+|k{2,}', ' emoji_vui ', text)

        # Domain-specific marker retained as a token; assumes downstream vocabulary can handle underscores
        text = re.sub(r'3///|3que|3\s*que', ' phản_động ', text)

        # Emphasis markers normalized to explicit tokens; avoids losing intent when stripping punctuation
        text = re.sub(r'\?{2,}', ' dấu_hỏi_gắt ', text)
        text = re.sub(r'!{2,}', ' dấu_chấm_than_gắt ', text)
        text = re.sub(r'\.{3,}', ' dấu_ba_chấm ', text)

        return text

    def remove_special_chars(self, text: str) -> str:
        """
        Retain basic sentence punctuation and underscores; strip non-informative symbols.
        Preserves custom tokens (e.g., emoji_vui) and sentence delimiters to reduce semantic loss.
        """
        return re.sub(r'[^\w\s\dàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,?!_]', ' ', text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_repeating_chars(self, text: str) -> str:
        """
        Reduce character runs >=3 to a single char to mitigate exaggerated spelling without corrupting valid words.
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def run(self, text: str) -> str:
        # Order matters: map semantics before case/cleanup to avoid losing signal
        text = self.replace_special_tokens(text)
        text = self.to_lower(text)
        text = self.normalize_repeating_chars(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        return text