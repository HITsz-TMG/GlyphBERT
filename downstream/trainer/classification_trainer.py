import torch
from downstream.models.classification_model import ClassifierBERT
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch import nn
import time
import os, sys
import re
import nltk
import math
import json

SEED = 5
torch.manual_seed(SEED)
class ClassifierTrainer:

    def __init__(self, train_dataset, dev_dataset, test_dataset, model: ClassifierBERT,
                 config=None, save_root=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        self.config = config
        self.epoch = config['epoch']
        self.batch_size = config['batch_size']
        self.batch_expand_times = config['batch_expand_times']
        self.lr = config['lr']

        self.save_root = save_root

        self.train_dataloader = train_dataset.get_dataloader(shuffle=True, batch_size=self.batch_size)
        self.dev_dataloader = dev_dataset.get_dataloader(shuffle=False, batch_size=self.batch_size)
        self.test_dataloader = test_dataset.get_dataloader(shuffle=False, batch_size=self.batch_size)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = Adam(optimizer_grouped_parameters, lr=self.lr, weight_decay=config.get('weight_decay', 0))
        one_epoch_backward_step_num = math.ceil(len(train_dataset) / self.batch_size)
        total_step = one_epoch_backward_step_num * self.epoch // self.batch_expand_times
        warm_up_step = config['warm_up'] if config['warm_up'] >= 1 else int(total_step * config['warm_up'])
        print("warm-up/total-step: {}/{}".format(warm_up_step, total_step))
        self.schedule = get_linear_schedule_with_warmup(
            self.optimizer, num_training_steps=total_step, num_warmup_steps=warm_up_step)

        self.config = config

        if config['parallel'] == 'data_parallel':
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

        self.save_name_list = []

        self.total_steps = total_step
        self.steps_eval = int(len(train_dataset) / (self.batch_size * self.batch_expand_times) * config['steps_eval'])
        self.start_eval_epoch = config['start_eval_epoch']
        self.current_steps = 0
        self.max_acc = float('-inf')

        self.use_res2bert = config.get("use_res2bert", False)
        self.model.use_res2bert = self.use_res2bert

    @torch.no_grad()
    def eval(self, eval_type="dev"):
        self.model.eval()
        if eval_type == "dev":
            iterator_bar = self.dev_dataloader
        else:
            iterator_bar = self.test_dataloader
        correct_num = 0
        total = 0
        for batch in iterator_bar:
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            outputs = self.model(**batch)
            logits = outputs[1]
            _, preds = logits.max(dim=-1)
            correct = (batch['labels'] == preds).sum().item()
            correct_num += correct
            total += len(preds)
        acc = correct_num / total

        return acc

    @torch.no_grad()
    def nli_predict_test(self):
        self.model.eval()
        result_dict = {"predict": []}
        cnt = 0
        for batch in tqdm(self.test_dataloader):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            cnt += 1

            cur = {
                "id": cnt
            }

            input_tokens = self.train_dataset.convert_ids_to_tokens(batch['input_ids'][0].tolist())
            cur['原文本'] = " ".join(input_tokens)
            outputs = self.model(**batch)
            logits = outputs[1]
            _, preds = logits.max(dim=-1)
            label = batch['labels'].item()

            cur['预测的标签'] = preds.item()
            cur['真实的标签'] = label

            result_dict['predict'].append(cur)

        json.dump(result_dict, open("./{}-NLI-result.json".format(self.cnn_name), 'w', encoding='utf-8'),
                  ensure_ascii=False)

    def model_select_save(self, dev_acc, epoch):
        if dev_acc > self.max_acc:
            self.max_acc = dev_acc
            test_acc = self.eval(eval_type="test")
            # self.save_state_dict(filename="{}-epoch{}-DevACC{:.5f}-TestACC{:.5f}.pt".format(
            #     self.config['dataset_name'], epoch, dev_acc, test_acc)
            # )
            return test_acc
        else:
            return None

    def train_epoch(self, epoch):
        self.model.train()
        if epoch < self.start_eval_epoch:
            iterator_bar = tqdm(self.train_dataloader)
        else:
            iterator_bar = self.train_dataloader
        loss_sum = 0.0
        step_num = len(iterator_bar)
        self.optimizer.zero_grad()
        if self.start_eval_epoch == epoch:
            self.current_steps = 0
        for step, batch in enumerate(iterator_bar):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            outputs = self.model(**batch)
            loss = outputs[0]
            # iterator_bar.set_description("EPOCH[{}] LOSS[{:.5f}]".format(epoch, loss.item()))
            loss_sum += loss.item()
            loss.backward()
            if ((step + 1) % self.batch_expand_times) == 0:
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
                self.current_steps += 1

                if epoch >= self.start_eval_epoch and (self.current_steps % self.steps_eval) == 0:
                    print("# EPOCH[{}]STEP[{}]-EVAL ".format(epoch, step), end=" ")
                    dev_acc = self.eval(eval_type='dev')
                    print(dev_acc)
                    test_acc = self.model_select_save(dev_acc, epoch)
                    if test_acc is not None:
                        print("Dev-Acc[{:.5f}] Test-Acc[{:.5f}]  Model-Saved".format(dev_acc, test_acc))
                    else:
                        print("dev acc[{:.5f}]".format(dev_acc))

        self.optimizer.step()
        self.optimizer.zero_grad()
        avg_loss = loss_sum / step_num
        return avg_loss

    def train(self):
        print("\nStart Training\n")
        for epoch in range(1, self.epoch + 1):
            avg_loss = self.train_epoch(epoch)
            print("EPOCH[{}] AVG_LOSS[{:.5f}]".format(epoch, avg_loss))

    def save_state_dict(self, filename="test.pt", max_save_num=1):
        save_path = os.path.join(self.save_root, filename)
        if self.config['parallel'] == 'data_parallel':
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

        check_list = []
        for i in self.save_name_list:
            if os.path.exists(i):
                check_list.append(i)
        if len(check_list) == max_save_num:
            del_file = check_list.pop(0)
            os.remove(del_file)
        self.save_name_list = check_list
        self.save_name_list.append(save_path)
