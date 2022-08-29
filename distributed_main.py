from config import config
from src.GlyphCNN import AddBertResPos3
from src.GlyphDataset import GlyphDataset
from src.GlyphBERT import GlyphBERT
from src.Trainer import Trainer
import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed
from main import load_existing_bert

if __name__ == '__main__':
    world_size = len(list(config['device'].split(',')))
    local_rank = config['local_rank']
    config['world_size'] = world_size
    torch.cuda.set_device(local_rank)

    train_dataset = GlyphDataset(config)

    model = GlyphBERT(config=config, glyph_embedding=AddBertResPos3)
    # model = load_existing_bert(model)
    if config.get("state_dict", None) is not None:
        print("Load state dict {}".format(config['state_dict']))
        model = model.to('cpu')
        model.load_state_dict(torch.load(config['state_dict'], map_location='cpu')['model'])

    model = model.to('cuda')
    distributed.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    trainer = Trainer(train_dataset, model, config=config)
    trainer.train()
