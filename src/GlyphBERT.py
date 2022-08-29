import json
from abc import ABC

import torch
from torch import nn
from src.hugging_face_v2 import (
    BertModel,
    BertConfig,
    BertOnlyNSPHead,
    BertOnlyMLMHead,
    BertPooler,
)


class GlyphBERT(nn.Module, ABC):
    def __init__(self, config=None, glyph_embedding=None):
        super().__init__()
        self.config = config
        bert_config_path = config['bert_config_path']
        self.bert_config = json.load(open(bert_config_path, "r"))
        self.vocab_size = self.bert_config['vocab_size']
        self.hidden_size = self.bert_config['hidden_size']
        self.bert_config = BertConfig.from_dict(self.bert_config)
        self.bert: BertModel = BertModel(config=self.bert_config, add_pooling_layer=False)
        self.use_res2bert = self.config['use_res2bert']#cnn residual
        if self.use_res2bert:
            print("\n Use Res2Bert \n")
        self.cnn_and_embed_mat = config.get("cnn_and_embed_mat", False)
        if self.cnn_and_embed_mat:
            print("\n Hybrid Glyph And ID Embedding \n")
            self.embedding_matrix = nn.Embedding(config['vocab_size'], embedding_dim=768, padding_idx=0)
            self.fusion_linear = nn.Linear(768 * 2, 768)

        self.cnn_embedding = glyph_embedding(config)

        self.pooler = BertPooler(self.bert_config)
        self.nsp = BertOnlyNSPHead(self.bert_config)
        self.mlm = BertOnlyMLMHead(self.bert_config)
        self.loss_fct = nn.CrossEntropyLoss()

    def glyph_bert_sequence_outputs(self, input_ids=None, token_type_ids=None, image_input=None,
                                    attention_mask=None, unique_ids=None):
        if unique_ids is not None:
            unique_embeds, first_res_embeds, second_res_embeds = self.cnn_embedding(
                input_ids=unique_ids.unsqueeze(dim=0),
                image_input=image_input,
            )
            unique_embeds = unique_embeds.squeeze(dim=0)
            glyph_map = torch.zeros((self.vocab_size, self.hidden_size)).to(input_ids.device)
            if self.use_res2bert:
                first_res_embeds = first_res_embeds.squeeze(dim=0)
                second_res_embeds = second_res_embeds.squeeze(dim=0)
                first_res_map = torch.zeros((self.vocab_size, self.hidden_size)).to(input_ids.device)
                second_res_map = torch.zeros((self.vocab_size, self.hidden_size)).to(input_ids.device)
            else:
                first_res_map, second_res_map = None, None
            for idx, i in enumerate(unique_ids.view(-1).tolist()):
                glyph_map[i] = unique_embeds[idx]
                if self.use_res2bert:
                    first_res_map[i] = first_res_embeds[idx]
                    second_res_map[i] = second_res_embeds[idx]
            input_embeds = torch.embedding(weight=glyph_map, indices=input_ids, padding_idx=0)
            if self.use_res2bert:
                first_res = torch.embedding(weight=first_res_map, indices=input_ids, padding_idx=0)
                second_res = torch.embedding(weight=second_res_map, indices=input_ids, padding_idx=0)
            else:
                first_res, second_res = None, None
        else:
            input_embeds, first_res, second_res = self.cnn_embedding(
                input_ids=input_ids,
                image_input=image_input,
            )

        if self.cnn_and_embed_mat:
            normal_embedding = self.embedding_matrix(input_ids)
            input_embeds = self.fusion_linear(torch.cat((input_embeds, normal_embedding), dim=-1))

        outputs = self.bert(
            input_ids=None,
            inputs_embeds=input_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            ##
            first_block_result=first_res,
            second_block_result=second_res,
            ##
        )
        sequence_output = outputs[0]

        return sequence_output

    def forward(self, input_ids=None, token_type_ids=None, labels=None, image_input=None,
                next_sentence_label=None, attention_mask=None, unique_ids=None):

        sequence_output = self.glyph_bert_sequence_outputs(
            input_ids=input_ids, token_type_ids=token_type_ids, image_input=image_input,
            attention_mask=attention_mask, unique_ids=unique_ids
        )

        mlm_logits = self.mlm(sequence_output)
        mlm_loss = self.loss_fct(mlm_logits.view(-1, self.bert_config.vocab_size), labels.view(-1))

        if self.config['add_nsp_task']:
            pooled_output = self.pooler(sequence_output)
            nsp_logits = self.nsp(pooled_output)
            nsp_loss = self.loss_fct(nsp_logits.view(-1, 2), next_sentence_label.view(-1))
        else:
            nsp_loss = None

        return {
            "loss": mlm_loss if nsp_loss is None else mlm_loss + nsp_loss,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
            # "embeddings": input_embeds,
            # "input_ids":input_ids,
        }

# if __name__ == '__main__':
#     transform_to_tensor = transforms.ToTensor()
#     image = Image.open("../data/bmp/22.bmp")
#     image_tensor = transform_to_tensor(image)
#     vocab = dict()
#     with open("../data/vocab_bmp.txt", 'r', encoding='utf-8') as reader:
#         for idx, i in enumerate(reader.readlines()):
#             vocab[idx] = i.strip()
