import os
import argparse
import json

root_path = "/home/wxy/GlyphBERT"
data_path = "/home/wxy/GlyphBERT/pretrain_data"


def get_path(p, is_data=False):
    if is_data:
        return os.path.join(data_path, p)
    else:
        return os.path.join(root_path, p)


config = {
    # env config
    "device": "0,1",  # "4,5,6,7"
    "parallel": None,  # "data_parallel" "DDP" None
    "local_rank": None,

    # training config
    "batch_per_card": 4,
    "batch_expand_times": 64,
    "epoch": 20,
    "warm_up": 10000,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "batch": None,  # batch will be calculate based on batch_per_card and batch_expand_times
    "glyph_map": True,  # False True
    "dataloader_workers": 4,

    # model config
    "CNN_name": "AddBertResPos3",
    "use_res2bert": True,
    "cnn_and_embed_mat": True,
    "add_nsp_task": True,
    "state_dict": ".pretrained_model/save/AddBertResPos3-epoch11-loss0.85704.pt",
    "board_path": "WikiFromEpoch4",

    # path config
    "bmp_path": get_path("data/bmp", is_data=True),
    "vocab_path": get_path("data/vocab_bmp.txt", is_data=True),
    "bert_config_path": get_path("pretrained_model/config.json"),
    "pretrained_data_name": "dupe4wiki",  # one_wiki DEBUG overflow dupe4wiki
}

parser = argparse.ArgumentParser()

# env config
parser.add_argument('--device', default=config.get('device'), type=str, required=False)
parser.add_argument('--parallel', default=config.get('parallel'), type=str, required=False)

# training config
parser.add_argument('--batch_per_card', default=config.get('batch_per_card'), type=int, required=False)
parser.add_argument('--batch_expand_times', default=config.get('batch_expand_times'), type=int, required=False)
parser.add_argument('--epoch', default=config.get('epoch'), type=int, required=False)
parser.add_argument('--warm_up', default=config.get('warm_up'), type=int, required=False)
parser.add_argument('--lr', default=config.get('lr'), type=float, required=False)
parser.add_argument('--glyph_map', default=config.get('glyph_map'), type=bool, required=False)

# model config
parser.add_argument('--CNN_name', default=config.get('CNN_name'), type=str, required=False)
parser.add_argument('--use_res2bert', default=config.get('use_res2bert'), type=bool, required=False)
parser.add_argument('--cnn_and_embed_mat', default=config.get('cnn_and_embed_mat'), type=bool, required=False)
parser.add_argument('--state_dict', default=config.get('state_dict'), type=str, required=False)
parser.add_argument('--add_nsp_task', default=config.get('add_nsp_task'), type=bool, required=False)

# path config
parser.add_argument('--pretrained_data_name', default=config.get('pretrained_data_name'), type=str, required=False)

# DDP
parser.add_argument('--local_rank', default=config.get('local_rank'), type=int, required=False)

args = vars(parser.parse_args())

cards_num = len(list(args['device'].split(',')))
batch_per_card = args['batch_per_card']
if args['parallel'] != "DDP":
    config['batch_size'] = cards_num * batch_per_card
else:
    config['batch_size'] = batch_per_card

for k in args.keys():
    if k not in config.keys():
        print("add new config key: {}={}".format(k, args[k]))
    config[k] = args[k]

if config.get('sentence_path') is None:
    vocab_size = 18612
    if config['pretrained_data_name'] == "13w":
        config['data_path'] = "./data/sentence_pair_with_mask_data_large.pkl"
    elif config['pretrained_data_name'] == "100w":
        config['data_path'] = "./data/sentence_pair_2_1.pkl"
    elif config['pretrained_data_name'] == "DEBUG":
        config['data_path'] = "./data/sentence_pair_with_mask_data.pkl"
    elif config['pretrained_data_name'] == "all":
        all_data_path_list = [
            "./data/all-dupe4-2021-4-2/{}.pkl".format(i) for i in range(13)
        ]
        all_data_path_list += [
            "./data/4-22-data/sentence_pair/THUC_836070_{}.pkl".format(i) for i in range(14)
        ]
        all_data_path_list += [
            "./data/4-22-data/sentence_pair/sogou0103data406500_{}.pkl".format(i) for i in range(3)
        ]
        all_data_path_list += [
            "./data/4-22-data/sentence_pair/QAbaike409600_{}.pkl".format(i) for i in range(3)
        ]
        config['data_path'] = all_data_path_list
    elif config['pretrained_data_name'] == '300w':
        config['data_path'] = ["./data/4-18-300w/{}.pkl".format(i) for i in range(3)]
    elif config['pretrained_data_name'] == "dupe4wiki":
        all_data_path_list = [
            "./data/dupe4wiki-10-13/{}.pkl".format(i) for i in range(13)
        ]
        config['data_path'] = all_data_path_list
    elif config['pretrained_data_name'] == "dupe4wiki16":
        all_data_path_list = [
            "./data/for16gpus/{}.pkl".format(i) for i in range(16)
        ]
        config['data_path'] = all_data_path_list
    elif config['pretrained_data_name'] == "one_wiki":
        config['data_path'] = "./data/dupe4wiki-10-13/0.pkl"

else:
    if input("\n Build sentence pair, input[y]: ") != 'y':
        exit(-1)
    config.update({
        "data_path": "./data/2021-04-18-sentence-pair-300w.pkl"
    })
    vocab_size = 18612

if isinstance(config['data_path'], str):
    config['data_path'] = [get_path(config['data_path'], is_data=True)]
else:
    for idx in range(len(config['data_path'])):
        config['data_path'][idx] = get_path(config['data_path'][idx], is_data=True)

config['vocab_size'] = vocab_size
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

# recover the previous training state
if config.get('state_dict'):
    import torch

    state_dict = torch.load(config['state_dict'], map_location='cpu')
    training_state = state_dict['training_state']
    prev_config = state_dict['config']
    config['lr'] = training_state['last_lr']
    if type(prev_config['warm_up']) is float:
        warm_up_step = training_state['total_optimize_step'] * prev_config['warm_up']
    else:
        warm_up_step = config['warm_up']
    assert training_state['optimize_step'] >= warm_up_step
    config['warm_up'] = 0
    config['rest_optimize_step'] = training_state['total_optimize_step'] - training_state['optimize_step']
    config['training_state'] = training_state

print("")
print("-" * 12 + "Config" + "-" * 12)
for k, v in config.items():
    if isinstance(v, list):
        if len(v) == 0:
            print(k, v)
        else:
            print(k, ':')
            for i in v:
                print("\t{}".format(i))
    else:
        print("{}: {}".format(k, v))
print("-" * 30)

# import random
# import pickle
# from tqdm import tqdm
#
# all_data = []
# for i in config['data_path']:
#     print("load {}".format(i))
#     all_data.extend(pickle.load(open(i, 'rb')))
# random.shuffle(all_data)
# random.shuffle(all_data)
# random.shuffle(all_data)
# each_size = 1000000
# cnt = 0
# for i in tqdm(range(0, len(all_data), each_size), total=len(all_data) // each_size + 1):
#     end = min(len(all_data), i + each_size)
#     pickle.dump(
#         all_data[i: end],
#         open("/mnt/inspurfs/user-fs/zhaoyu/pretrain_data/data/dupe4wiki-10-13/{}.pkl".format(cnt), 'wb')
#     )
#     cnt += 1
