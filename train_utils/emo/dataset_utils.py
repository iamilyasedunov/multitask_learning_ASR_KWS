import os
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Tuple, Union, List, Callable, Optional
import pathlib
import torchaudio.transforms as T
from datetime import datetime

class EmoDataset(Dataset):

    def __init__(
            self,
            transform: Optional[Callable] = None,
            csv: Optional[pd.DataFrame] = None
    ):
        self.transform = transform
        self.csv = csv

    def __getitem__(self, index: int):
        instance = self.csv.iloc[index]
        path2wav = instance['wav']
        wav, sr = torchaudio.load(path2wav)

        if sr != 16000:
            wav = T.Resample(orig_freq=sr, new_freq=16000)(wav)

        wav = wav.sum(dim=0)

        if self.transform:
            wav = self.transform(wav)
        return {
            'wav': wav,
            'label': instance['label']
        }

    def __len__(self):
        return len(self.csv)