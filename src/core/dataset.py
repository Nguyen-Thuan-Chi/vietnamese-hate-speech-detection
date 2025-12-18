import torch
from torch.utils.data import Dataset
from typing import List
from src.core.dtos import HateSpeechSample
from transformers import PreTrainedTokenizer


class HateSpeechDataset(Dataset):
    def __init__(self, data: List[HateSpeechSample],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int = 128):
        """
        data: Danh sách các DTO đã được làm sạch (clean text).
        tokenizer: Bộ dịch của PhoBERT (biến chữ thành số).
        max_len: Độ dài tối đa của 1 câu (nếu dài hơn thì cắt, ngắn hơn thì thêm số 0).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Mapping nhãn: Chắc chắn rằng label đang là số int (0 hoặc 1)
        # Nếu label đang là string "1", code dưới sẽ tự ép kiểu
        pass

    def __len__(self):
        # Trả về tổng số lượng mẫu dữ liệu
        return len(self.data)

    def __getitem__(self, index):
        # 1. Lấy ra 1 mẫu dữ liệu tại vị trí index
        sample = self.data[index]
        text = str(sample.text)

        # Xử lý nhãn: Chuyển từ string/int sang kiểu LongTensor của Torch
        # (Lưu ý: Nếu nhãn là None thì để mặc định là 0 - dùng khi chạy dự đoán)
        label = int(sample.label) if sample.label is not None else 0

        # 2. Dùng Tokenizer để mã hóa text
        # encoding là một cái từ điển chứa: input_ids, attention_mask
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Thêm token đặc biệt [CLS] ở đầu, [SEP] ở cuối
            max_length=self.max_len,  # Giới hạn độ dài
            padding='max_length',  # Nếu ngắn quá thì thêm số 0 vào đuôi (padding)
            truncation=True,  # Nếu dài quá thì cắt bớt
            return_attention_mask=True,  # Trả về mặt nạ (để model biết đâu là chữ thật, đâu là số 0 padding)
            return_tensors='pt',  # Trả về dạng PyTorch Tensor
        )

        # 3. Trả về kết quả đóng gói
        # flatten() dùng để duỗi thẳng vector (ví dụ [[1, 2]] thành [1, 2]) cho gọn
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }