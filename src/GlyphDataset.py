from torch.utils import data
import pickle
from torch.nn.utils.rnn import pad_sequence
import os
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from itertools import chain
from tqdm import tqdm
from transformers.models.albert import AlbertForQuestionAnswering

class GlyphDataset(data.Dataset):
    def __init__(self, config, loaded_data=None):

        vocab_path = config['vocab_path']
        bmp_path = config['bmp_path']
        data_path = config['data_path']

        self.config = config
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
        self.data = []

        if config['local_rank'] is not None:
            local_rank = config['local_rank']
            world_size = config['world_size']
            per_size = len(data_path) // world_size
            rest_data_path = data_path[per_size * world_size:]
            data_path = data_path[local_rank * per_size: (local_rank + 1) * per_size]
            print("\nrank {} will load data from {}\n and rest data path {} \n".format(local_rank, data_path,
                                                                                       rest_data_path))
            for i in data_path:
                cur_data = pickle.load(open(i, 'rb'))
                self.data.extend(cur_data)
                print("rank {} load from {}, len {}".format(local_rank, i, len(cur_data)))
            for i in rest_data_path:
                cur_data = pickle.load(open(i, 'rb'))
                per_size = len(cur_data) // world_size
                self.data.extend(cur_data[local_rank * per_size: (local_rank + 1) * per_size])
                print("rank {} load Rest Data from {}[{}:{}], len {}".format(
                    local_rank, i, local_rank * per_size, (local_rank + 1) * per_size, len(cur_data)))

            print("rank {} load sentence pair num {}".format(local_rank, len(self)))

        elif loaded_data is None:
            if isinstance(data_path, list):
                for i in data_path:
                    cur_data = pickle.load(open(i, 'rb'))
                    self.data.extend(cur_data)
                    print("load from {}, len {}".format(i, len(cur_data)))
                print("load sentence pair num {}".format(len(self)))
            else:
                self.data = pickle.load(open(data_path, 'rb'))
                print("load sentence pair num {}".format(len(self)))
        else:
            self.data = loaded_data

        print(len(self.data))

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
        if self.data[0].get("unique_ids") is None:
            self.prepare_dataset()
        # self.prepare_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def stack_image(self, input_ids_list):
        return torch.stack(
            [self.bmp[t] for t in input_ids_list], dim=0
        )

    def prepare_dataset(self):
        for idx in tqdm(range(len(self.data)), desc="Prepare Dataset"):
            self.data[idx]['input_ids'] = self.data[idx]['input_ids'][:512]
            self.data[idx]['token_type_ids'] = self.data[idx]['token_type_ids'][:512]
            self.data[idx]['labels'] = self.data[idx]['labels'][:512]
            self.data[idx]['unique_ids'] = list(set(self.data[idx]['input_ids']))
        # each_size = 1000000
        # cnt = 0
        # for i in tqdm(range(0, len(self.data), each_size), total=len(self.data) // each_size + 1):
        #     end = min(len(self.data), i + each_size)
        #     pickle.dump(
        #         self.data[i: end],
        #         open("/mnt/inspurfs/user-fs/zhaoyu/pretrain_data/data/dupe4wiki-10-12-unique/{}.pkl".format(cnt), 'wb')
        #     )
        #     cnt += 1

    def get_dataloader(self, batch_size=8, shuffle=False, num_workers=0, sampler=None):
        pad_idx = self.pad_idx
        not_predict_tag = self.not_predict_tag

        def collate_fn(batch):
            # unique_ids = torch.tensor([list(set(chain(*[i['unique_ids'] for i in batch])))])
            input_ids = [torch.tensor(i['input_ids']) for i in batch]
            token_type_ids = [torch.tensor(i['token_type_ids']) for i in batch]
            labels = [torch.tensor(i['labels']) for i in batch]
            next_sentence_label = torch.tensor([i['next_sentence_label'] for i in batch])

            # padding
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=pad_idx)
            labels = pad_sequence(labels, batch_first=True, padding_value=not_predict_tag)

            # mask padding的部分 0: mask
            attention_mask = (input_ids != pad_idx).long()

            ret_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "labels": labels,
                "next_sentence_label": next_sentence_label,
                "attention_mask": attention_mask,
                # "unique_ids": unique_ids,
            }
            if self.config['glyph_map'] is False:
                ret_data["image_input"] = self.stack_image(input_ids.view(-1).tolist())
            else:
                unique_ids = torch.tensor(list(set(chain(*[i['unique_ids'] for i in batch]))))
                ret_data["unique_ids"] = unique_ids
                ret_data["image_input"] = self.stack_image(unique_ids.view(-1).tolist())

            return ret_data

        return data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
