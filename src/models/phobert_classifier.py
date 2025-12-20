import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class HateSpeechClassifier(nn.Module):
    def __init__(self, model_name: str = "vinai/phobert-base-v2", n_classes: int = 2):
        super(HateSpeechClassifier, self).__init__()

        # Load PhoBERT backbone for Vietnamese; weights must align with tokenizer used upstream
        self.bert = AutoModel.from_pretrained(model_name)

        # Classification head applied on pooled sentence representation; dropout regularizes fine-tuning
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward expects tokenized input aligned with PhoBERT. Returns raw logits for downstream loss/metrics.
        """
        # Use pooled output for sentence-level classification; return_dict=False to retain tuple API
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        output = self.drop(pooled_output)
        return self.out(output)