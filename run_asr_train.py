import torchaudio
from ruamel.yaml import YAML

yaml = YAML()
from torch import optim

from logger import *
import os
import torch.utils.data as data
from preprocessing.data_processing import *
from model.model_asr import SpeechRecognitionModel
from train_utils.asr.train import IterMeter, Trainer
from utils.asr.decoder import TextTransform
from tqdm.auto import tqdm

def main(config):
    config_asr = config["asr"]
    writer = get_writer(config["wandb"])
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device(config_asr["device"])
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    if not os.path.isdir(config_asr["data_path"]):
        os.makedirs(config_asr["data_path"])

    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )

    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

    text_transform = TextTransform()

    train_dataset = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["train_url"], download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["test_url"], download=True)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config_asr['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x, train_audio_transforms, valid_audio_transforms, text_transform, 'train'),
                                   **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=config_asr['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x, train_audio_transforms, valid_audio_transforms, text_transform, 'valid'),
                                  **kwargs)

    model = SpeechRecognitionModel(
        config_asr['n_cnn_layers'], config_asr['n_rnn_layers'], config_asr['rnn_dim'],
        config_asr['n_class'], config_asr['n_feats'], config_asr['stride'], config_asr['dropout']
    ).to(device)
    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), config_asr['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config_asr['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=config_asr['epochs'],
                                              anneal_strategy='linear')
    iter_meter = IterMeter()
    trainer = Trainer(model, device, train_loader, test_loader, criterion, optimizer, scheduler, 0,
                      iter_meter, writer, config_asr["log_step"], config)
    if os.path.exists(config['asr']['checkpont_path']):
        trainer.load_checkpoint()
    else:
        print("Attention! Checkpoint path is not exists, model train from starting initialization.")

    for _ in tqdm(range(1, config_asr["epochs"] + 1), desc='main_loop', total=config_asr["epochs"]):
        trainer.train()
        trainer.test()


if __name__ == "__main__":
    config_path = "other/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f) #, Loader=yaml.Loader)
    main(config)
