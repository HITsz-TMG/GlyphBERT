
from config import config
from src.hugging_face_v2 import BertPreTrainedModel, BertConfig
import time
import os
import random
from utils.log_utils import ZYLog
from src.GlyphCNN import AddBertResPos3, GlyphEmbedding
from src.GlyphDataset import GlyphDataset
from src.GlyphBERT import GlyphBERT
from src.Trainer import Trainer
import torch

if not os.path.exists("running_log"):
    os.mkdir("running_log")

log = ZYLog(os.path.join("running_log", time.strftime("%m-%d-%H-%M")))
log.write_config(config)


def load_existing_bert(model):
    bert_path = "/mnt/inspurfs/user-fs/zhaoyu/pretrained_model/chinese_bert_base/pytorch_model.bin"
    bert_vocab_path = "/mnt/inspurfs/user-fs/zhaoyu/pretrained_model/chinese_bert_base/vocab.txt"

    glyph_vocab = dict()
    glyph_token2idx = dict()
    with open(config['vocab_path'], 'r', encoding='utf-8') as reader:
        for idx, i in enumerate(reader.readlines()):
            glyph_vocab[idx] = i.strip()
            glyph_token2idx[i.strip()] = idx

    bert_vocab = dict()
    with open(bert_vocab_path, 'r', encoding='utf-8') as reader:
        for idx, i in enumerate(reader.readlines()):
            bert_vocab[idx] = i.strip()

    new_embedding = torch.zeros(len(glyph_vocab), 768)
    bert_state_dict = torch.load(bert_path, map_location='cpu')

    for bert_token_id, bert_token in bert_vocab.items():
        glyph_token_id = glyph_token2idx.get(bert_token)
        if glyph_token_id is None:
            continue
        new_embedding[glyph_token_id] = bert_state_dict['bert.embeddings.word_embeddings.weight'].data[bert_token_id]
    del bert_state_dict['bert.embeddings.word_embeddings.weight']
    bert_state_dict['embedding_matrix.weight'] = new_embedding
    new_state_dict = dict()
    for k, v in bert_state_dict.items():
        if k.startswith('bert.pooler'):
            new_state_dict[k[5:]] = v
        elif k.startswith('cls.seq_relationship'):
            new_state_dict['nsp' + k[3:]] = v
        elif k.startswith('cls.predictions'):
            continue  # vocabs' size are not equal
            # new_state_dict['mlm' + k[3:]] = v
        else:
            new_state_dict[k] = v
    false_params = model.load_state_dict(new_state_dict, strict=False)
    print(false_params)
    assert all(i.startswith('mlm.predictions') or
               i.startswith('cnn_embedding') or
               i == 'bert.embeddings.word_embeddings.weight' for i in false_params.missing_keys)
    assert len(false_params.unexpected_keys) == 0
    return model


def main():
    log.write_log("build model")
    model = GlyphBERT(config=config, glyph_embedding=AddBertResPos3)


    params_num = 0
    for n, p in model.named_parameters():
        if "cnn_embedding" in n:
            params_num += p.numel()
    print(params_num)

    # model = load_existing_bert(model)

    if config.get("state_dict", None) is not None:
        log.write_log("load state dict {}".format(config['state_dict']))
        print("Load state dict {}".format(config['state_dict']))
        model = model.to('cpu')
        checkpoint = torch.load(config['state_dict'],map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    log.write_log("read training dataset")
    train_dataset = GlyphDataset(config)
    log.write_log("loaded training dataset")

    random.shuffle(train_dataset.data)

    log.write_log("build trainer")
    trainer = Trainer(train_dataset, model, config=config)

    log.write_log("training start")
    trainer.train()

    log.write_log("training end")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.write_log(repr(e))
        print(RuntimeError)
        raise RuntimeError
