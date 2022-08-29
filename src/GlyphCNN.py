import torch
import torch.nn as nn


class GlyphEmbedding(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def forward(self, *args, **kwargs) -> tuple:
        raise NotImplementedError


class AddBertResPos3(GlyphEmbedding):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.config = config
        print("\n AddBertResPos3 \n")
        # 1*48*48
        self.conv_pre = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 48-3+1+2=48
        self.res_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=9, stride=1)  # 48-9+1=40
        # 64*42*42
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=9, stride=1)  # 40-9+1=32
        # 64*36*36
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=9, stride=1)  # 32-9+1=24
        # self.conv1_3 + self.res_1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32*12*12
        self.conv_pre_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 64*12*12
        self.res_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  # 12-3+1=10
        # 64*8*8
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  # 10-3+1=8
        # 64*6*6
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  # 8-3+1=6
        # self.res_2 + conv2_3
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128*3*3
        self.fc1 = nn.Linear(1152, 768)
        self.dropout = nn.Dropout(0.1)

        if self.config['use_res2bert']:
            print("use res2bert")
            self.first_linear = nn.Linear(9216, 768)
            self.second_linear = nn.Linear(1152, 768)
        self.act = torch.tanh

    def forward(self, input_ids, image_input=None):
        input_ids_shape = input_ids.shape
        x = image_input
        x = torch.relu(self.conv_pre(x))
        res_1 = self.res_1(x)
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = torch.relu(self.conv1_3(x) + res_1)
        x = self.maxpool1(x)
        first_block_result = x.view(x.shape[0], -1)
        x = torch.relu(x)
        x = torch.relu(self.conv_pre_2(x))
        res_2 = self.res_2(x)
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = torch.relu(self.conv2_3(x) + res_2)
        x = self.maxpool2(x)
        second_block_result = x.view(x.shape[0], -1)
        x = torch.tanh(self.fc1(second_block_result))
        x = self.dropout(x)
        x = x.view(input_ids_shape[0], -1, 768)

        first_block_info, second_block_info = None, None
        if self.config['use_res2bert']:
            first_block_info = self.act(self.first_linear(first_block_result)).view(input_ids_shape[0], -1, 768)
            second_block_info = self.act(self.second_linear(second_block_result)).view(input_ids_shape[0], -1, 768)

        return x, first_block_info, second_block_info
