import torch
from src.GlyphBERT import GlyphBERT
from src.GlyphDataset import GlyphDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import time
import os
import math
from torch.utils.tensorboard import SummaryWriter
import shutil


# tensorboard --logdir=tensorboard --bind_all

class Trainer:

    def __init__(self, train_dataset: GlyphDataset, model: GlyphBERT, config=None, train_dataloader=None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.model = model
        self.config = config
        self.epoch = config['epoch']
        self.batch_size = config['batch_size']
        self.batch_expand_times = config['batch_expand_times']
        self.lr = config['lr']
        self.cnn_embedding_name = config['CNN_name']
        self.local_rank = config['local_rank']
        self.is_local_0 = self.local_rank is None or self.local_rank == 0
        print("Trainer rank {}, is_local_0:{}".format(self.local_rank, self.is_local_0))

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=0.01,
                               correct_bias=False, eps=1e-6, betas=(0.9, 0.98))

        if config.get("rest_optimize_step"):
            total_optimize_step = config['rest_optimize_step']
        else:
            one_epoch_backward_step_num = math.ceil(len(train_dataset) / self.batch_size)
            total_optimize_step = one_epoch_backward_step_num * self.epoch // self.batch_expand_times

        self.optimize_step = 0
        self.backward_step = 0
        self.total_optimize_step = total_optimize_step
        self.last_epoch_avg_loss = None

        # BERT: warmup 10000
        warm_up_step = config['warm_up'] if config['warm_up'] >= 1 else int(total_optimize_step * config['warm_up'])
        print("warm-up/total-step: {}/{}".format(warm_up_step, total_optimize_step))
        self.schedule = get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=self.total_optimize_step,
            num_warmup_steps=warm_up_step,
        )

        self.data_sampler = None
        self.train_dataloader = None

        self.train_dataloader = train_dataset.get_dataloader(
            shuffle=True, batch_size=self.batch_size,
            num_workers=config['dataloader_workers']
        )

        if self.device == 'cuda':
            self.model = self.model.cuda()

        if config['parallel'] == 'data_parallel':
            self.model = nn.DataParallel(self.model)

        self.save_name_list = []
        if not os.path.exists("./save"):
            os.mkdir("save")

        if self.is_local_0:
            self.board = SummaryWriter('./running_log/{}'.format(config['board_path']))
        else:
            self.board = None

    def get_training_state(self):
        return {
            "last_lr": self.schedule.get_last_lr()[0],
            "backward_step": self.backward_step,
            "optimize_step": self.optimize_step,
            "total_optimize_step": self.total_optimize_step,
            "last_epoch_avg_loss": self.last_epoch_avg_loss,
        }

    def train_epoch(self, epoch):
        self.model.train()
        iterator_bar = tqdm(self.train_dataloader, ncols=100) if self.is_local_0 else self.train_dataloader
        loss_sum = 0.0
        step_num = len(iterator_bar)
        loss_cache = []
        cache_start_time = time.perf_counter()
        for batch in iterator_bar:
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            outputs = self.model(**batch)
            loss = outputs["loss"]

            if self.config['parallel'] == "data_parallel":
                loss = loss.mean()
            bar_description = "EPOCH[{}] LOSS[{:.5f}] OPTIMIZE[{}]".format(epoch, loss.item(), self.optimize_step)

            if self.is_local_0:
                iterator_bar.set_description(bar_description)

            loss_sum += loss.item()
            loss_cache.append(loss.item())
            loss /= self.config['batch_expand_times']
            loss.backward()
            self.backward_step += 1

            if self.backward_step % self.batch_expand_times == 0:
                self.optimizer.step()
                self.schedule.step()
                self.optimizer.zero_grad()
                self.optimize_step += 1

                if self.is_local_0:
                    self.board.add_scalar("loss", sum(loss_cache) / len(loss_cache), self.optimize_step)
                    self.board.add_scalar("time", time.perf_counter() - cache_start_time, self.optimize_step)
                loss_cache = []
                cache_start_time = time.perf_counter()

                if (self.optimize_step + 1) % 4000 == 0:
                    print("rank {} is alive, optimize_step {}".format(self.local_rank, self.optimize_step))
                    self.save_state_dict("time[{}]-step[{}].pt".format(
                        time.strftime("%m-%d-%H-%M"), self.optimize_step + 1))

        avg_loss = loss_sum / step_num

        return avg_loss

    def train(self):
        min_loss = float('inf')
        print("Start Training")
        # self.save_state_dict(filename='init_save[{}]'.format(time.strftime("%m-%d-%H-%M")), add_to_save_list=False)
        for epoch in range(1, self.epoch + 1):
            avg_loss = self.train_epoch(epoch)
            self.last_epoch_avg_loss = avg_loss
            print("--- Backward step {}, Current Optimize Step {}, Target Optimize_step {}".format(
                self.backward_step, self.optimize_step, self.total_optimize_step))
            print("--- EPOCH[{}] AVG_LOSS[{:.5f}] LR[{}]".format(epoch, avg_loss, self.schedule.get_last_lr()[0]))
            if avg_loss < min_loss:
                min_loss = avg_loss
                self.save_state_dict(filename="{}-epoch{}-loss{:.5f}.pt".format(
                    self.cnn_embedding_name, epoch, avg_loss), add_to_save_list=False)

        self.optimizer.zero_grad()

    def save_state_dict(self, filename="test.pt", max_save_num=3, add_to_save_list=True):
        if self.is_local_0 is False:
            return
        save_path = os.path.join("./save", filename)
        if self.config['parallel'] in ['data_parallel', 'DDP']:
            torch.save({"config": self.config,
                        "training_state": self.get_training_state(),
                        "model": self.model.module.state_dict()}, save_path)
        else:
            torch.save({"config": self.config,
                        "training_state": self.get_training_state(),
                        "model": self.model.state_dict()}, save_path)
        if not add_to_save_list:
            return

        check_list = []
        if save_path in self.save_name_list:
            print("overlay")
            return
        for i in self.save_name_list:
            if os.path.exists(i):
                check_list.append(i)
        if len(check_list) == max_save_num:
            del_file = check_list.pop(0)
            os.remove(del_file)
        self.save_name_list = check_list
        self.save_name_list.append(save_path)


if __name__ == '__main__':
    pass
