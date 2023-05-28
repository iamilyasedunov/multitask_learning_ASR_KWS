import torchaudio
from ruamel.yaml import YAML

yaml = YAML()
from torch import optim
import torch
from train_utils.asr_kws.dataset_utils import prepare_datasets, AsrMultitaskDataSet, MultitaskCollator, get_transforms
from train_utils.asr_kws.train import Trainer
from logger import *
import os
import torch.utils.data as data
from train_utils.asr.train import IterMeter
from model.model_asr_kws import MultitaskModel
from tqdm.auto import tqdm


def main(config):
    config_asr = config["asr"]
    config_kws = config['multitask']['kws_cmd']
    config_emo = config['multitask']['emo']
    writer = get_writer(config["wandb"])
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device(config_asr["device"])
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    if not os.path.isdir(config_asr["data_path"]):
        os.makedirs(config_asr["data_path"])

    train_set_kws, val_set_kws, train_set_asr, val_set_asr, train_set_emo, val_set_emo = prepare_datasets(config_asr, config_kws, config_emo)
    print(f"train_set_kws: {len(train_set_kws)}, val_set_kws: {len(val_set_kws)}, train_set_asr: {len(train_set_asr)}, val_set_asr: {len(val_set_asr)}, train_set_emo: {len(train_set_emo)}, val_set_emo: {len(val_set_emo)}")
    train_multitask_set = AsrMultitaskDataSet(train_set_asr, train_set_kws, train_set_emo)
    val_multitask_set = AsrMultitaskDataSet(val_set_asr, val_set_kws, val_set_emo)

    transforms = get_transforms()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_multitask_set,
                                   batch_size=config['multitask']['batch_size'],
                                   shuffle=True,
                                   collate_fn=MultitaskCollator(transforms, 'train'),
                                   **kwargs)
    val_loader = data.DataLoader(dataset=val_multitask_set,
                                 batch_size=config['multitask']['val_batch_size'],
                                 shuffle=False,
                                 collate_fn=MultitaskCollator(transforms, 'val'),
                                 **kwargs)

    multitask_model = MultitaskModel(config_asr['n_cnn_layers'], config_asr['n_rnn_layers'], config_asr['rnn_dim'],
                                     config_asr['n_class'], config_asr['n_feats'], config_emo['num_class'], config_asr['stride'],
                                     config_asr['dropout']
                                     ).to(device)
    optimizer = optim.AdamW(multitask_model.parameters(), config['multitask']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
                                              #max_lr=config['multitask']['learning_rate'],
                                              #steps_per_epoch=int(len(train_loader)),
                                              #epochs=config['multitask']['epochs'],
                                              #anneal_strategy='linear')
    iter_meter = IterMeter()

    trainer = Trainer(multitask_model, device, train_loader, val_loader, optimizer, scheduler, 0,
                      iter_meter, writer, config['multitask']["log_step"], config)

    if os.path.exists(config['multitask']['checkpont_path']):
        trainer.load_checkpoint()
    else:
        print("Attention! Checkpoint path is not exists, model train from starting initialization.")

    for epoch in tqdm(range(1, config['multitask']["epochs"] + 1), desc='main_loop',
                      total=config['multitask']["epochs"]):
        trainer.train()
        trainer.test()
        if epoch != 1:
            trainer.save_checkpoint(epoch)
        trainer.scheduler.step(trainer.last_test_loss)

if __name__ == "__main__":
    config_path = "other/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f)  # , Loader=yaml.Loader)
    main(config)
