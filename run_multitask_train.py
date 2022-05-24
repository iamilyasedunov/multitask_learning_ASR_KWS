import torchaudio
from ruamel.yaml import YAML

yaml = YAML()
from torch import optim
import torch
from train_utils.asr_kws.dataset_utils import prepare_datasets, AsrMultitaskDataSet, MultitaskCollator
from logger import *
import os



def main(config):
    config_asr = config["asr"]
    config_kws = config['multitask']['kws']
    writer = get_writer(config["wandb"])
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device(config_asr["device"])
    if not os.path.isdir(config_asr["data_path"]):
        os.makedirs(config_asr["data_path"])

    train_set_kws, val_set_kws, train_set_asr, val_set_asr = prepare_datasets(config_asr, config_kws)

    train_multitask_set = AsrMultitaskDataSet(train_set_asr, train_set_kws)
    val_multitask_set = AsrMultitaskDataSet(val_set_asr, val_set_kws)



    print(len(train_set), len(dataset_asr))


if __name__ == "__main__":
    config_path = "other/default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f) #, Loader=yaml.Loader)
    main(config)
