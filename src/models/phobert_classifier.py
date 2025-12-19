import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class HateSpeechClassifier(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base-v2", n_classes: int = 2):
        super(HateSpeechClassifier, self).__init__()

        # 1. Load "Ông giáo sư" PhoBERT từ kho của VinAI
        # AutoModel sẽ tải toàn bộ trọng số (weights) đã được train sẵn
        self.bert = AutoModel.from_pretrained(model_name)

        # 2. Tạo cái "Đầu" phân loại (Classifier Head)
        # PhoBERT base output ra vector kích thước 768
        # Dropout: Kỹ thuật giúp model đỡ bị học vẹt (Overfitting) - Ngẫu nhiên tắt 10% nơ-ron
        self.drop = nn.Dropout(p=0.3)

        # Lớp Linear cuối cùng: Biến vector 768 -> vector 2 (Tương ứng 2 lớp: Sạch/Độc)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Hàm này mô tả dòng chảy của dữ liệu qua mạng neuron.
        Input: Câu chữ đã số hóa.
        Output: Điểm số (logits) cho từng nhãn.
        """

        # Bước 1: Cho dữ liệu đi qua PhoBERT
        # output[0] là hidden states của tất cả token
        # output[1] (pooler_output) là vector đại diện cho TOÀN BỘ CÂU (thường dùng cho classification)
        # Tuy nhiên với PhoBERT/BERT, người ta hay dùng output[1] hoặc last_hidden_state[:, 0, :] (Token [CLS])

        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Trả về tuple cho dễ lấy index
        )

        # Bước 2: Đi qua lớp Dropout (để thử thách model)
        output = self.drop(pooled_output)

        # Bước 3: Đi qua lớp Linear cuối cùng để ra kết quả
        return self.out(output)