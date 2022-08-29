from abc import ABC
import torch
from torch import nn
from src.GlyphBERT import GlyphBERT


class ClassifierBERT(nn.Module, ABC):
    def __init__(self, bert: GlyphBERT, cnn, class_num=2):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn = cnn
        self.bert = bert
        input_size = 768
        self.classify = nn.Linear(input_size, class_num)
        # self.classify = nn.Sequential(
        #     nn.Linear(input_size, input_size * 2),
        #     nn.Tanh(),
        #     nn.LayerNorm(input_size * 2, eps=1e-12),
        #     nn.Linear(input_size * 2, class_num)
        # )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, labels=None, attention_mask=None,
                token_type_ids=None, image_input=None, unique_ids=None):

        sequence_output = self.bert.glyph_bert_sequence_outputs(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            image_input=image_input,
            attention_mask=attention_mask,
            unique_ids=unique_ids
        )
        cls_sequence = sequence_output[:, 0]
        logits = self.classify(cls_sequence)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return logits
