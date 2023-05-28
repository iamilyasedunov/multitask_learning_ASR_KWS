import torchaudio, os
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from augmentations.augs_creation import AugsCreation
from utils.asr.decoder import TextTransform
import pandas as pd
from sklearn.utils import resample
from preprocessing.data_processing import *
from train_utils.kws.dataset_utils import DatasetDownloader, SpeechCommandDataset
from datetime import datetime

class AsrThreeTaskDataSet(Dataset):
    def __init__(self, asr_dataset: Dataset, kws_cmd_dataset: Dataset):
        self.asr_dataset = asr_dataset
        self.kws_cmd_dataset = kws_cmd_dataset

    def __len__(self):
        return max(len(self.asr_dataset), len(self.kws_cmd_dataset))

    def __getitem__(self, index: int):
        # start = datetime.now()
        wav_asr, _, transcript, _, _, _ = self.asr_dataset.__getitem__(index % len(self.asr_dataset))
        # print(f"get_item asr: {(datetime.now() - start).total_seconds()}")
        # start = datetime.now()
        kws_cmd_data = self.kws_cmd_dataset.__getitem__(index % len(self.kws_cmd_dataset))
        wav_cmd_kws, keyword_kws, label_kws, label_cmd = kws_cmd_data['wav'], kws_cmd_data['keyword'], kws_cmd_data['label'], kws_cmd_data['label_cmd']
        # print(f"get_item kws: {(datetime.now() - start).total_seconds()}")
        # start = datetime.now()
        # print(f"get_item emo: {(datetime.now() - start).total_seconds()}")
        return (wav_asr, transcript, wav_cmd_kws, keyword_kws, label_kws, label_cmd)


class MultitaskCollator:
    def __init__(self, transforms: dict, phase: str):
        self.transforms = transforms
        self.phase = phase

    def preprocessing(self, data_item):
        wav_asr, transcript, wav_cmd_kws, keyword_kws, label_kws, label_cmd = data_item
        # asr preprocessing
        spec_asr = self.transforms['spec_asr'][self.phase](wav_asr)
        spec_asr = spec_asr.squeeze(0).transpose(0, 1)
        # print(f"asr: {spec_asr.size()}")
        label_asr = torch.Tensor(self.transforms['text'].text_to_int(transcript.lower()))
        input_length_asr = spec_asr.shape[0] // 2
        label_length_asr = len(label_asr)
        # kws preprocessing
        spec_kws_cmd = self.transforms['spec_kws_cmd'][self.phase](wav_cmd_kws)
        spec_kws_cmd = spec_kws_cmd.squeeze(0).transpose(0, 1)
        # print(f"kws: {spec_kws.size()}")
        return [spec_asr, label_asr, input_length_asr, label_length_asr, spec_kws_cmd, label_cmd, label_kws]

    def __call__(self, data):
        dict_keys = ['spec_asr', 'label_asr', 'input_length_asr', 'label_length_asr', 'spec_kws_cmd', 'label_cmd', 'label_kws']
        dict_data = {key: [] for key in dict_keys}
        for item in data:
            preprocessed_data = self.preprocessing(item)
            for key, prep_item in zip(dict_keys, preprocessed_data):
                dict_data[key].append(prep_item)

        spec_asr = torch.nn.utils.rnn.pad_sequence(dict_data['spec_asr'], batch_first=True).unsqueeze(1).transpose(2, 3)
        spec_kws_cmd = torch.nn.utils.rnn.pad_sequence(dict_data['spec_kws_cmd'], batch_first=True).unsqueeze(1).transpose(2, 3)

        # print(f"dict_data['spec_asr']: {dict_data['spec_asr'][0].size()}, spec_asr: {spec_asr[0].size()}")
        # print(f"dict_data['spec_kws']: {dict_data['spec_kws'][0].size()}, spec_kws: {spec_kws[0].size()}")
        # print(f"dict_data['spec_emo']: {dict_data['spec_emo'][0].size()}") # , spec_asr: {spec_emo.sizes()}")
        label_asr = torch.nn.utils.rnn.pad_sequence(dict_data['label_asr'], batch_first=True)
        label_cmd = torch.Tensor(dict_data['label_cmd']).long()
        label_kws = torch.Tensor(dict_data['label_kws']).long()
        return spec_asr, label_asr, dict_data['input_length_asr'], dict_data['label_length_asr'], spec_kws_cmd, label_cmd, label_kws


def prepare_datasets(config_asr, config_kws):
    if os.path.exists(config_kws['train_path']) and os.path.exists(config_kws['val_path']):
        train_df_kws = pd.read_csv(config_kws['train_path'])
        val_df_kws = pd.read_csv(config_kws['val_path'])
    else:
        _ = DatasetDownloader(config_kws['key_word'])
        dataset_kws = SpeechCommandDataset(path2dir='speech_commands', keywords=config_kws['key_word'])
        data_len_scaled = int(len(dataset_kws) * config_kws['kws_data_percent']) if config_kws['kws_data_percent'] else len(dataset_kws)
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
        train_df_kws = pd.concat([df_majority, df_minority_upsampled])
        val_df_kws = dataset_kws.csv.iloc[val_indexes].reset_index(drop=True)
        # train_df_kws.to_csv('/home/sedunov_ia/project_mtl/multitask_learning_ASR_KWS/other/train_kws_cmd.csv', index=False)
        # val_df_kws.to_csv('/home/sedunov_ia/project_mtl/multitask_learning_ASR_KWS/other/val_kws_cmd.csv', index=False)
    train_set_kws_cmd = SpeechCommandDataset(csv=train_df_kws, transform=AugsCreation())
    val_set_kws_cmd = SpeechCommandDataset(csv=val_df_kws)

    train_set_asr = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["train_url"], download=True)
    print(f"len: asr : {len(train_set_asr)}")
    val_set_asr = torchaudio.datasets.LIBRISPEECH(config_asr["data_path"], url=config_asr["test_url"], download=True)


    return train_set_kws_cmd, val_set_kws_cmd, train_set_asr, val_set_asr


def get_transforms():
    dict_transforms = {'spec_kws_cmd': {}, 'spec_asr': {}, 'spec_emo': {}}
    dict_transforms['spec_kws_cmd']['train'] = nn.Sequential(
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
    dict_transforms['spec_kws_cmd']['val'] = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128,
                n_fft=400,
                win_length=400,
                hop_length=160,
            )

    dict_transforms['text'] = TextTransform()
    return dict_transforms
