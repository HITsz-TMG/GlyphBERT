from torch.utils import data
from tqdm import tqdm
import torch
import random
import pickle, os, csv
from torch.nn.utils.rnn import pad_sequence
import json
from PIL import Image, ImageOps
from torchvision import transforms
from itertools import chain


class ClassifierDataset(data.Dataset):
    def __init__(self, vocab_path, seq_len, data_path, corpus_path=None, preprocessing=False, dataset_name=None,
                 json_data=False, config=None):
        self.idx2token = dict()
        with open(vocab_path, 'r', encoding='utf-8') as reader:
            for idx, i in enumerate(reader.readlines()):
                self.idx2token[idx] = i.strip()
        self.vocab = self.idx2token

        self.token2idx = dict([(token, idx) for idx, token in self.idx2token.items()])
        self.pad_idx = self.token2idx['[PAD]']
        self.unk_idx = self.token2idx['[UNK]']
        self.cls_idx = self.token2idx['[CLS]']
        self.sep_idx = self.token2idx['[SEP]']
        self.mask_idx = self.token2idx['[MASK]']
        self.special_token_num = 5
        self.not_predict_tag = -100
        self.seq_len = seq_len

        self.dataset_name = dataset_name

        if preprocessing:
            self.data = self._preprocessing(corpus_path)
            pickle.dump(self.data, open(data_path, 'wb'))
        else:
            if json_data:
                self.data = []
                tmp_data = json.load(open(data_path, 'r', encoding='utf-8'))
                for item in tmp_data:
                    self.data.append({
                        "input_ids": self.convert_tokens_to_ids(item['input_ids']),
                        "label": item['label']
                    })
            else:
                self.data = pickle.load(open(data_path, 'rb'))
        bmp_path = config['bmp_path']
        self.bmp = dict()
        transform_to_tensor = transforms.ToTensor()
        for idx, token in self.vocab.items():
            token_img_path = os.path.join(bmp_path, "{}.bmp".format(idx))
            token_img = Image.open(token_img_path)
            token_img = token_img.convert('L')
            token_img = ImageOps.invert(token_img)
            token_img = token_img.convert('1')
            self.bmp[idx] = (transform_to_tensor(token_img))

        self.channel_x = torch.tensor(
            [[(i - 24) * 0.02 for i in range(0, 48)] for _ in range(0, 48)]
        ).unsqueeze(dim=0)
        self.channel_y = torch.tensor(
            [[(i - 24) * 0.02 for i in range(0, 48)] for _ in range(0, 48)]
        ).transpose(0, 1).unsqueeze(dim=0) * -1

        for key in self.bmp:
            self.bmp[key] = torch.cat((self.bmp[key], self.channel_x, self.channel_y))

        self.prepare_dataset()

    def prepare_dataset(self):
        for idx in tqdm(range(len(self.data)), desc="Prepare Dataset"):
            self.data[idx]['unique_ids'] = list(set(self.data[idx]['input_ids']))

    def convert_tokens_to_ids(self, tokens):
        return [self.token2idx.get(i, self.unk_idx) for i in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[i] for i in ids]

    def _preprocessing(self, corpus_path):

        if self.dataset_name == "hotel":
            self.split_and_process_hotel()
            print("hotel 预处理结束")
            exit(0)

        def remove_space(t):
            t = t.replace(' ', '').replace("\n", '').replace('\t', '').replace('\u3000', '').replace('\u00A0', '')
            return [i for i in t]

        res_data = []
        with open(corpus_path, 'r', encoding='utf-8') as reader:
            raw_data = reader.readlines()

        raw_data = raw_data[1:]
        for line in raw_data:
            assert line[0] == '1' or line[0] == '0'
            label = int(line[0])
            text = line[1:].strip()
            text = remove_space(text)
            res_data.append({
                "input_ids": [self.cls_idx] + self.convert_tokens_to_ids(text)[:510] + [self.sep_idx],
                "label": label
            })

        return res_data

    def split_and_process_hotel(self):
        def remove_space(t):
            t = t.replace(' ', '').replace("\n", '').replace('\t', '').replace('\u3000', '').replace('\u00A0', '')
            return [i for i in t]

        pos_data = []
        with open("../hotel/pos.txt") as reader:
            for line in reader.readlines():
                label, text = line.split('   ')
                label = int(label.strip())
                text = remove_space(text.strip())
                assert label == 1
                pos_data.append({
                    "input_ids": [self.cls_idx] + self.convert_tokens_to_ids(text)[:510] + [self.sep_idx],
                    "label": 1
                })
        neg_data = []
        with open("../hotel/neg.txt") as reader:
            for line in reader.readlines():
                label, text = line.split('   ')
                label = int(label.strip())
                text = remove_space(text.strip())
                assert label == -1
                neg_data.append({
                    "input_ids": [self.cls_idx] + self.convert_tokens_to_ids(text)[:510] + [self.sep_idx],
                    "label": 0
                })

        random.shuffle(pos_data)
        random.shuffle(neg_data)

        train_pos = pos_data[:int(0.8 * len(pos_data))]
        dev_pos = pos_data[int(0.8 * len(pos_data)):int(0.9 * len(pos_data))]
        test_pos = pos_data[int(0.9 * len(pos_data)):]

        train_neg = neg_data[:int(0.8 * len(neg_data))]
        dev_neg = neg_data[int(0.8 * len(neg_data)):int(0.9 * len(neg_data))]
        test_neg = neg_data[int(0.9 * len(neg_data)):]

        train = train_pos + train_neg
        dev = dev_pos + dev_neg
        test = test_pos + test_neg

        pickle.dump(train, open("../data/hotel_train.pkl", "wb"))
        pickle.dump(dev, open("../data/hotel_test.pkl", "wb"))
        pickle.dump(test, open("../data/hotel_dev.pkl", "wb"))

    def stack_image(self, input_ids_list):
        return torch.stack(
            [self.bmp[t] for t in input_ids_list], dim=0
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        pad_idx = self.pad_idx

        def collate_fn(batch):
            input_ids = [torch.tensor(i['input_ids']) for i in batch]
            labels = torch.tensor([int(i['label']) for i in batch])
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            attention_mask = (input_ids != pad_idx).long()

            ret_data = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
            if 'unique_ids' in batch[0].keys():
                unique_ids = torch.tensor(list(set(chain(*[i['unique_ids'] for i in batch]))))
                ret_data["unique_ids"] = unique_ids
                ret_data["image_input"] = self.stack_image(unique_ids.view(-1).tolist())
            else:
                ret_data["image_input"] = self.stack_image(input_ids.view(-1).tolist())

            return ret_data

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
