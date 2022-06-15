import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from augmentations.augs_creation import AugsCreation
from utils.asr.decoder import TextTransform
import pandas as pd
from sklearn.utils import resample
from preprocessing.data_processing import *
from train_utils.kws.dataset_utils import DatasetDownloader, SpeechCommandDataset


class AsrMultitaskDataSet(Dataset):
    def __init__(self, asr_dataset: Dataset, kws_dataset: Dataset):
        self.asr_dataset = asr_dataset
        self.kws_dataset = kws_dataset

    def __len__(self):
        return max(len(self.asr_dataset), len(self.kws_dataset))

    def __getitem__(self, index: int):
        wav_asr, _, transcript, _, _, _ = self.asr_dataset.__getitem__(index % len(self.asr_dataset))
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
        spec_asr = self.transforms['spec_asr'][self.phase](wav_asr)
        spec_asr = spec_asr.squeeze(0).transpose(0, 1)
        # print(f"asr: {spec_asr.size()}")
        label_asr = torch.Tensor(self.transforms['text'].text_to_int(transcript.lower()))
        input_length_asr = spec_asr.shape[0] // 2
        label_length_asr = len(label_asr)
        # kws preprocessing
        spec_kws = self.transforms['spec_kws'][self.phase](wav_kws)
        spec_kws = spec_kws.squeeze(0).transpose(0, 1)
        # print(f"kws: {spec_kws.size()}")

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
    # upsampling
    df_majority = train_df[train_df.label == 0]
    df_minority = train_df[train_df.label == 1]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=42)  # reproducible results

    # Combine majority class with upsampled minority class
    train_df = pd.concat([df_majority, df_minority_upsampled])
    # print(train_df)
    val_df = dataset_kws.csv.iloc[val_indexes].reset_index(drop=True)
    train_set_kws = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set_kws = SpeechCommandDataset(csv=val_df)

    train_set_asr = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["train_url"], download=True)
    print(f"len: asr : {len(train_set_asr)}")
    val_set_asr = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["test_url"], download=True)
    return train_set_kws, val_set_kws, train_set_asr, val_set_asr


def get_transforms():
    dict_transforms = {'spec_kws': {}, 'spec_asr': {}}
    dict_transforms['spec_kws']['train'] = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128,
                                             n_fft=400,
                                             win_length=400,
                                             hop_length=160,
                                             ),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        #torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    dict_transforms['spec_asr']['train'] = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )
    dict_transforms['spec_asr']['val'] = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    dict_transforms['spec_kws']['val'] = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128,
                n_fft=400,
                win_length=400,
                hop_length=160,
            )

    dict_transforms['text'] = TextTransform()
    return dict_transforms
