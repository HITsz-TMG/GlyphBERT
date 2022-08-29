from classification_config import config
from downstream.trainer.classification_trainer import ClassifierTrainer
from downstream.dataset.classification_dataset import ClassifierDataset
from downstream.models.classification_model import ClassifierBERT
#from downstream.models.token_classification_model import TokenClassifierBERT
from src.GlyphBERT import GlyphBERT
import torch
# from src.hugging_face import BertModel, BertConfig, BertForPreTraining
from src.GlyphCNN import AddBertResPos3
from torch.nn import functional as F
def get_empty_model(exp_time=1):
    pretrained_model = GlyphBERT(config=config, glyph_embedding=AddBertResPos3)
    checkpoint = torch.load(config['pretrained_model_path'], map_location='cpu')
    pretrained_model.load_state_dict(checkpoint['model'])
    new_model = ClassifierBERT(pretrained_model, pretrained_model.cnn_embedding, class_num=config['num_labels'])
    if config.get("state_dict", None) is not None and exp_time == 1:
        new_model.load_state_dict(torch.load(config['state_dict'], map_location='cpu')['model'])
    return new_model


if __name__ == '__main__':
    SEED = config['seed']
    torch.manual_seed(SEED)
    train_dataset = ClassifierDataset(config['vocab_path'], 512,
                                      data_path=config['train_data_path'],
                                      preprocessing=config['preprocessing'],
                                      dataset_name=config['dataset_name'],
                                      json_data=config.get('json_data'), config=config)

    dev_dataset = ClassifierDataset(config['vocab_path'], 512,
                                    data_path=config['dev_data_path'],
                                    preprocessing=config['preprocessing'],
                                    dataset_name=config['dataset_name'],
                                    json_data=config.get('json_data'), config=config)

    test_dataset = ClassifierDataset(config['vocab_path'], 512,
                                     data_path=config['test_data_path'],
                                     preprocessing=config['preprocessing'],
                                     dataset_name=config['dataset_name'],
                                     json_data=config.get('json_data'), config=config)

    save_root = config['save_root']

    if config.get("task", None) is not None:
        model = get_empty_model()
        trainer = ClassifierTrainer(train_dataset, dev_dataset, test_dataset,
                                    model, config=config, save_root=save_root)
        acc = trainer.eval("test")
        print("test accuracy:{:.5f}".format(acc))
    elif config.get('exp_times', None) is None:
        model = get_empty_model()
        trainer = ClassifierTrainer(train_dataset, dev_dataset, test_dataset,
                                    model, config=config, save_root=save_root)
        #acc = trainer.classification_eval("test")
        trainer.train()
        acc = trainer.eval("test")
    else:
        for i in range(1, 1 + config['exp_times']):
            print("--- fine-tune times [{}]".format(i))
            trainer = ClassifierTrainer(train_dataset, dev_dataset, test_dataset,
                                        get_empty_model(exp_time=i), config=config, save_root=save_root)
            #acc = trainer.classification_eval("test")
            #acc = trainer.eval("test")
            trainer.train()
            acc = trainer.eval("test")
