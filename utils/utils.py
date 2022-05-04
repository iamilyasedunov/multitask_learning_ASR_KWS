import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict as edict
from thop import clever_format

# import wandb
import os, sys
import torch
from torch.utils.data import DataLoader
from torch import distributions

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from thop import profile
import torchaudio
from IPython import display as display_
import warnings

from torch.cuda.amp import autocast


# Helper functions


class Collator:
    def __call__(self, data):
        wavs = []
        labels = []

        for el in data:
            wavs.append(el['wav'])
            labels.append(el['label'])

        # torch.nn.utils.rnn.pad_sequence takes list(Tensors) and returns padded (with 0.0) Tensor
        wavs = pad_sequence(wavs, batch_first=True)
        labels = torch.Tensor(labels).long()
        return wavs, labels


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def prune_model_l1_unstructured(model, config_pruning_unstructured):
    for module in model.modules():
        for config_item in config_pruning_unstructured:
            layer_type = config_item["layer"]
            proportion = config_item["prob"]

            if isinstance(module, layer_type):
                print(layer_type, proportion)
                print(module)
                prune.l1_unstructured(module, 'weight', proportion)
                prune.remove(module, 'weight')
    return model


def prune_model_l1_structured(model, config_pruning_structured):
    for module in model.modules():
        for config_item in config_pruning_structured:
            layer_type = config_item["layer"]
            proportion = config_item["prob"]
            if isinstance(module, layer_type):
                print(layer_type, proportion)
                print(module)
                prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
                prune.remove(module, 'weight')
    return model


def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size_of_model_mb = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size_of_model_mb


def count_macs_params(model, loader, preprocessor, device):
    batch, labels = next(iter(loader))
    batch, labels = batch.to(device), labels.to(device)
    batch = preprocessor(batch)
    with HiddenPrints():
        warnings.filterwarnings("default")
        macs, params = profile(model, (batch,))
    param_size = next(model.parameters()).element_size()
    model_size_in_mb = (params * param_size) / (2 ** 20)
    return macs, params, round(model_size_in_mb, 3)


def get_model_info(model, loader, preprocessor, device):
    print(f"CRNN [conv] num params: {count_parameters(model.conv)}")
    print(f"CRNN [gru] num params    : {count_parameters(model.gru)}")
    print(f"CRNN [attn] num params    : {count_parameters(model.attention)}")
    print(f"CRNN [clsfr] num params    : {count_parameters(model.classifier)}")

    macs, params, model_size_in_mb = count_macs_params(
        model, loader, preprocessor, device)
    macs_pretty, params_pretty = clever_format([macs, params], "%.3f")
    size_model_mb = get_size_of_model(model)
    print(f"Num params      : {params}, ({params_pretty})")
    print(f"Named param     : {sum([(w != 0).sum() for _, w in model.named_parameters()])}")
    print(f"MACs            : {macs}, ({macs_pretty})")
    print(f"Model size in mb: {model_size_in_mb}, ({size_model_mb})")
    #for n, w in model.named_parameters():
    #    print(n, (w != 0).sum())


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)

    return model


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def make_config(key_word, config):
    config = edict(config)
    return config


# We should provide to WeightedRandomSampler _weight for every sample_; by default it is 1/len(target)

def get_sampler(target):
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])  # for every class count it's number of occ.
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def batch_data(data):
    wavs = []
    labels = []

    for el in data:
        wavs.append(el['utt'])
        labels.append(el['label'])

    # torch.nn.utils.rnn.pad_sequence takes list(Tensors) and returns padded (with 0.0) Tensor
    wavs = pad_sequence(wavs, batch_first=True)
    labels = torch.Tensor(labels).type(torch.long)
    return wavs, labels


# Quality measurment functions:
# FA - true: 0, model: 1
# FR - true: 1, model: 0

def count_FA_FR(preds, labels):
    FA = torch.sum(preds[labels == 0])
    FR = torch.sum(labels[preds == 0])

    # torch.numel - returns total number of elements in tensor
    return FA.item() / torch.numel(preds), FR.item() / torch.numel(preds)


def get_au_fa_fr(probs, labels):
    sorted_probs, _ = torch.sort(probs)
    sorted_probs = torch.cat((torch.Tensor([0]), sorted_probs, torch.Tensor([1])))
    labels = torch.cat(labels, dim=0)

    FAs, FRs = [], []
    for prob in sorted_probs:
        preds = (probs >= prob) * 1
        FA, FR = count_FA_FR(preds, labels)
        FAs.append(FA)
        FRs.append(FR)
    # plt.plot(FAs, FRs)
    # plt.show()

    # ~ area under curve using trapezoidal rule
    return -np.trapz(FRs, x=FAs)


def get_size_in_megabytes(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size = next(model.parameters()).element_size()
    return num_params, (num_params * param_size) / (2 ** 20)
