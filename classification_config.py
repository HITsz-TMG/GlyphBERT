import os
import argparse

root_path = "/mnt/inspurfs/user-fs/wxy/glyphCRM/" # your project pwd
data_path = "/mnt/inspurfs/user-fs/wxy/glyphCRM/"

def get_path(p, is_data=False):
    if is_data:
        return os.path.join(data_path, p)
    else:
        return os.path.join(root_path, p)


config = {
    #"device": "0",
    "seed": 2048, #random seed
    "device": "3", #gpu device
    "num_labels": 2, #classification labels
    "epoch": 10,
    "batch_size": 32,
    "batch_expand_times": 1,
    "warm_up": 0.2,
    "weight_decay": 0,
    "steps_eval": 0.2,
    "start_eval_epoch": 3,
    "exp_times": 3,
    "lr":3e-5,
    "bmp_path": get_path("data/bmp48/", is_data=True), #The preprocess bmp image path, preprocessed with a given vocab
    "vocab_path": get_path("data/vocab.txt", is_data=True),#The given vocab.
    #"vocab_path": get_path("data/vocab_bmp.txt", is_data=True),
    "bert_config_path": get_path("pretrained_model/config.json"),
    "preprocessing": False,
    "use_res2bert": True,
    "cnn_and_embed_mat": True,
    "parallel": None,
    "vocab_size": 18612,
    "json_data": True,
    #"save_root": "./downstream/save",
    "state_dict": None,  # state_dict is not pretrained_model_path
}

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=config.get('device'), type=str, required=False)
parser.add_argument('--epoch', default=config.get('epoch'), type=int, required=False)
parser.add_argument('--batch_size', default=config.get('batch_size'), type=int, required=False)
parser.add_argument('--batch_expand_times', default=config.get('batch_expand_times'), type=int, required=False)
parser.add_argument('--warm_up', default=config.get('warm_up'), type=float, required=False)
parser.add_argument('--weight_decay', default=config.get('weight_decay'), type=float, required=False)
parser.add_argument('--lr', default=config.get('lr'), type=float, required=False)
parser.add_argument('--steps_eval', default=config.get('steps_eval'), type=float, required=False)
parser.add_argument('--start_eval_epoch', default=config.get('start_eval_epoch'), type=int, required=False)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--exp_times', default=config.get('exp_times'), type=int, required=False)
parser.add_argument('--pretrained_model_path', type=str, required=True)
parser.add_argument('--use_res2bert', default=config.get('use_res2bert'), type=bool, required=False)
parser.add_argument('--cnn_and_embed_mat', default=config.get('cnn_and_embed_mat'), type=bool, required=False)
parser.add_argument('--state_dict', default=config.get('state_dict'), type=str, required=False)
parser.add_argument('--save_root', default=config.get('save_root'), type=str, required=False)
args = vars(parser.parse_args())

for k in args.keys():
    if k not in config.keys():
        print("add new config key: {}={}".format(k, args[k]))
    config[k] = args[k]

config['save_root']="./downstream/"+config['dataset_name']
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

if config['dataset_name'] == "chnsenti" and config['json_data']==True:
    config.update({
        "train_data_path": get_path("data/downstream_data/senti_train.json", is_data=True),
        "dev_data_path": get_path("data/downstream_data/senti_dev.json", is_data=True),
        "test_data_path": get_path("data/downstream_data/senti_test.json", is_data=True),
    })
# elif config['dataset_name'] == "hotel" and config['json_data']==True:
#     config.update({
#         "train_data_path": get_path("./data/classification_zy_data/hotel_train.json", is_data=True),
#         "dev_data_path": get_path("./data/classification_zy_data/hotel_dev.json", is_data=True),
#         "test_data_path": get_path("./data/classification_zy_data/hotel_test.json", is_data=True),
#     })
# elif config['dataset_name'] == "onlinesenti_cls" and config['json_data']==True:
#     config.update({
#         "train_data_path": get_path("./data/classification_zy_data/onlinesenti_train.json", is_data=True),
#         "dev_data_path": get_path("./data/classification_zy_data/onlinesenti_dev.json", is_data=True),
#         "test_data_path": get_path("./data/classification_zy_data/onlinesenti_test.json", is_data=True),
#     })
# elif config['dataset_name'] == "cnews" and config['json_data']==True:
#     config.update({
#         "train_data_path": get_path("./data/classification_zy_data/cnews_train.json", is_data=True),
#         "dev_data_path": get_path("./data/classification_zy_data/cnews_dev.json", is_data=True),
#         "test_data_path": get_path("./data/classification_zy_data/cnews_test.json", is_data=True),
#     })
# elif config['dataset_name'] == "toutiao" and config['json_data']==True:
#     config.update({
#         "train_data_path": get_path("./data/classification_zy_data/toutiao_train.json", is_data=True),
#         "dev_data_path": get_path("./data/classification_zy_data/toutiao_dev.json", is_data=True),
#         "test_data_path": get_path("./data/classification_zy_data/toutiao_test.json", is_data=True),
#     })
# elif config['dataset_name'] == "medical" and config['json_data']==True:
#     config.update({
#         "train_data_path": get_path("./data/classification_zy_data/medical_train.json", is_data=True),
#         "dev_data_path": get_path("./data/classification_zy_data/medical_dev.json", is_data=True),
#         "test_data_path": get_path("./data/classification_zy_data/medical_test.json", is_data=True),
#     })
# elif config['dataset_name'] == "afqmc":
#     config.update({
#         "train_data_path": get_path("Baseline/afqmc_public/afqmc_train_preprocess.pkl", is_data=True),
#         "dev_data_path": get_path("Baseline/afqmc_public/afqmc_dev_preprocess.pkl", is_data=True),
#         "test_data_path": get_path("Baseline/afqmc_public/afqmc_dev_preprocess.pkl", is_data=True),
#     })
#
# elif config['dataset_name'] == "iflytek":
#     config.update({
#         "train_data_path": get_path("Baseline/iflytek_public/iflytek_train_preprocess.pkl", is_data=True),
#         "dev_data_path": get_path("Baseline/iflytek_public/iflytek_dev_preprocess.pkl", is_data=True),
#         "test_data_path": get_path("Baseline/iflytek_public/iflytek_dev_preprocess.pkl", is_data=True),
#     })
#
# elif config['dataset_name'] == "tnews":
#     config.update({
#         "train_data_path": get_path("Baseline/tnews_public/tnews_train_preprocess.pkl", is_data=True),
#         "dev_data_path": get_path("Baseline/tnews_public/tnews_dev_preprocess.pkl", is_data=True),
#         "test_data_path": get_path("Baseline/tnews_public/tnews_dev_preprocess.pkl", is_data=True),
#     })


print(config)
print("")

# ./save/AddBertResPos3-epoch2-loss1.03873.pt

""" 参数记录

onlinesenti:
--lr=3e-5 --epoch=10 --steps_eval=0.1 --start_eval_epoch=2 --batch_size=8 --warm_up=0.2

hotel:
--lr=2e-5 --epoch=10 --steps_eval=0.2 --start_eval_epoch=3 --batch_size=32 --warm_up=0.1

chnsenti:
--lr=2e-5 --epoch=10 --steps_eval=0.2 --start_eval_epoch=3 --batch_size=32 --warm_up=0.1

"""