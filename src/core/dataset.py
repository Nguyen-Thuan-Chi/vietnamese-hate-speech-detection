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
        Dataset for sentence-level classification; expects preprocessed text and integer labels.
        Tokenization is performed lazily per item to balance memory and simplicity; adjust if throughput demands.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Labels are expected as 0/1 strings or ints; downstream loss requires contiguous integer classes
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        text = str(sample.text)

        # Default to label 0 when absent to support inference-only datasets
        label = int(sample.label) if sample.label is not None else 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }