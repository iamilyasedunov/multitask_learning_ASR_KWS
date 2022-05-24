import os
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Dict, Tuple, Union, List, Callable, Optional
import pathlib
from augmentations.augs_creation import AugsCreation

from preprocessing.data_processing import *
from train_utils.kws.dataset_utils import DatasetDownloader, SpeechCommandDataset


class AsrMultitaskDataSet(Dataset):
    def __init__(self, asr_dataset: Dataset, kws_dataset: Dataset):
        self.asr_dataset = asr_dataset
        self.kws_dataset = kws_dataset

    def __getitem__(self, index: int):
        wav_asr, _, transcript, _, _, _ = self.asr_dataset.__getitem__(index)
        kws_data = self.kws_dataset.__getitem__(index % len(self.kws_dataset))
        wav_kws, keyword_kws, label_kws = kws_data['wav'], kws_data['keyword'], kws_data['label']
        return (wav_asr, transcript, wav_kws, keyword_kws, label_kws)


class MultitaskCollator:
    def __init__(self, transforms: dict, phase: str):
        self.transforms = transforms
        self.phase = phase

    def preprocessing(self, data_item):
        wav_asr, transcript, wav_kws, keyword_kws, label_kws = data_item
        # asr preprocessing
        spec_asr = self.transforms['spec'][self.phase](wav_asr).squeeze(0).transpose(0, 1)
        label_asr = torch.Tensor(self.transforms['text'].text_to_int(transcript.lower()))
        input_length_asr = spec_asr.shape[0] // 2
        label_length_asr = len(label_asr)
        # kws preprocessing
        spec_kws = self.transforms['spec'][self.phase](wav_kws).squeeze(0).transpose(0, 1)
        return [spec_asr, label_asr, input_length_asr, label_length_asr, spec_kws, label_kws]

    def __call__(self, data):
        dict_keys = ['spec_asr', 'label_asr', 'input_length_asr', 'label_length_asr', 'spec_kws', 'label_kws']
        dict_data = {key: [] for key in dict_keys}
        for item in data:
            preprocessed_data = self.preprocessing(item)
            for key, prep_item in zip(dict_keys, preprocessed_data):
                dict_data[key].append(prep_item)

        spec_asr = torch.nn.utils.rnn.pad_sequence(dict_data['spec_asr'], batch_first=True).unsqueeze(1).transpose(2, 3)
        spec_kws = torch.nn.utils.rnn.pad_sequence(dict_data['spec_kws'], batch_first=True).unsqueeze(1).transpose(2, 3)
        label_asr = torch.nn.utils.rnn.pad_sequence(dict_data['label_asr'], batch_first=True)
        label_kws = torch.Tensor(dict_data['label_kws']).long()
        return spec_asr, label_asr, dict_data['input_length_asr'], dict_data['label_length_asr'], spec_kws, label_kws


def prepare_datasets(config_asr, config_kws):
    _ = DatasetDownloader(config_kws['key_word'])
    dataset_kws = SpeechCommandDataset(
        path2dir='speech_commands', keywords=config_kws['key_word']
    )
    data_len_scaled = int(len(dataset_kws) * config_kws['kws_data_percent']) if config_kws['kws_data_percent'] else len(
        dataset_kws)
    print(data_len_scaled)
    indexes = torch.randperm(data_len_scaled)
    train_indexes = indexes[:int(data_len_scaled * config_kws['train_test_split_percent'])]
    val_indexes = indexes[int(data_len_scaled * config_kws['train_test_split_percent']):]
    train_df = dataset_kws.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset_kws.csv.iloc[val_indexes].reset_index(drop=True)
    train_set_kws = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set_kws = SpeechCommandDataset(csv=val_df)

    train_set_asr = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["train_url"], download=True)
    val_set_asr = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["test_url"], download=True)
    return train_set_kws, val_set_kws, train_set_asr, val_set_asr
