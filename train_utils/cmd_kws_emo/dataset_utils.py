import torchaudio, os
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from augmentations.augs_creation import AugsCreation
from utils.asr.decoder import TextTransform
import pandas as pd
from sklearn.utils import resample
from preprocessing.data_processing import *
from train_utils.kws.dataset_utils import DatasetDownloader, SpeechCommandDataset
from train_utils.emo.dataset_utils import EmoDataset
from datetime import datetime

class CmdKwsEmoDataSet(Dataset):
    def __init__(self, kws_cmd_dataset: Dataset, emo_dataset: Dataset):
        self.kws_cmd_dataset = kws_cmd_dataset
        self.emo_dataset = emo_dataset

    def __len__(self):
        return max(len(self.kws_cmd_dataset), len(self.emo_dataset))

    def __getitem__(self, index: int):
        # start = datetime.now()
        # print(f"get_item asr: {(datetime.now() - start).total_seconds()}")
        # start = datetime.now()
        kws_cmd_data = self.kws_cmd_dataset.__getitem__(index % len(self.kws_cmd_dataset))
        wav_cmd_kws, keyword_kws, label_kws, label_cmd = kws_cmd_data['wav'], kws_cmd_data['keyword'], kws_cmd_data['label'], kws_cmd_data['label_cmd']
        # print(f"get_item kws: {(datetime.now() - start).total_seconds()}")
        # start = datetime.now()
        emo_data = self.emo_dataset.__getitem__(index % len(self.emo_dataset))
        wav_emo, label_emo = emo_data['wav'], emo_data['label']
        # print(f"get_item emo: {(datetime.now() - start).total_seconds()}")
        return (wav_cmd_kws, label_cmd, keyword_kws, label_kws, wav_emo, label_emo)


class MultitaskCollator:
    def __init__(self, transforms: dict, phase: str):
        self.transforms = transforms
        self.phase = phase

    def preprocessing(self, data_item):
        wav_cmd_kws, label_cmd, keyword_kws, label_kws, wav_emo, label_emo = data_item
        # asr preprocessing
        # kws preprocessing
        spec_kws_cmd = self.transforms['spec_kws_cmd'][self.phase](wav_cmd_kws)
        spec_kws_cmd = spec_kws_cmd.squeeze(0).transpose(0, 1)
        # print(f"kws: {spec_kws.size()}")
        spec_emo = self.transforms['spec_emo'][self.phase](wav_emo)
        spec_emo = spec_emo.squeeze(0).transpose(0, 1) # .transpose(0, 1)
        return [spec_kws_cmd, label_cmd, label_kws, spec_emo, label_emo]

    def __call__(self, data):
        dict_keys = ['spec_kws_cmd', 'label_cmd', 'label_kws', 'spec_emo', 'label_emo']
        dict_data = {key: [] for key in dict_keys}
        for item in data:
            preprocessed_data = self.preprocessing(item)
            for key, prep_item in zip(dict_keys, preprocessed_data):
                dict_data[key].append(prep_item)

        spec_kws_cmd = torch.nn.utils.rnn.pad_sequence(dict_data['spec_kws_cmd'], batch_first=True).unsqueeze(1).transpose(2, 3)
        spec_emo = torch.nn.utils.rnn.pad_sequence(dict_data['spec_emo'], batch_first=True).unsqueeze(1).transpose(2, 3)

        # print(f"dict_data['spec_asr']: {dict_data['spec_asr'][0].size()}, spec_asr: {spec_asr[0].size()}")
        # print(f"dict_data['spec_kws']: {dict_data['spec_kws'][0].size()}, spec_kws: {spec_kws[0].size()}")
        # print(f"dict_data['spec_emo']: {dict_data['spec_emo'][0].size()}") # , spec_asr: {spec_emo.sizes()}")

        label_cmd = torch.Tensor(dict_data['label_cmd']).long()
        label_kws = torch.Tensor(dict_data['label_kws']).long()
        label_emo = torch.Tensor(dict_data['label_emo']).long()
        return spec_kws_cmd, label_cmd, label_kws, spec_emo, label_emo


def prepare_datasets(config_kws, config_emo):
    if os.path.exists(config_kws['train_path']) and os.path.exists(config_kws['val_path']):
        train_df_kws = pd.read_csv(config_kws['train_path'])
        val_df_kws = pd.read_csv(config_kws['val_path'])
    else:
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
        train_df_kws = pd.concat([df_majority, df_minority_upsampled])
        val_df_kws = dataset_kws.csv.iloc[val_indexes].reset_index(drop=True)
        # train_df_kws.to_csv('/home/sedunov_ia/project_mtl/multitask_learning_ASR_KWS/other/train_kws_cmd.csv', index=False)
        # val_df_kws.to_csv('/home/sedunov_ia/project_mtl/multitask_learning_ASR_KWS/other/val_kws_cmd.csv', index=False)

    train_set_kws_cmd = SpeechCommandDataset(csv=train_df_kws, transform=AugsCreation())
    val_set_kws_cmd = SpeechCommandDataset(csv=val_df_kws)

    train_df_emo = pd.read_csv(config_emo['train_path'], sep='|')
    val_df_emo = pd.read_csv(config_emo['val_path'], sep='|')

    train_set_emo = EmoDataset(csv=train_df_emo, transform=AugsCreation())
    val_set_emo = EmoDataset(csv=val_df_emo)

    return train_set_kws_cmd, val_set_kws_cmd, train_set_emo, val_set_emo


def get_transforms():
    dict_transforms = {'spec_kws_cmd': {}, 'spec_emo': {}}
    dict_transforms['spec_kws_cmd']['train'] = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128,
                                             n_fft=400,
                                             win_length=400,
                                             hop_length=160,
                                             ),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        #torchaudio.transforms.TimeMasking(time_mask_param=35)
    )

    dict_transforms['spec_emo']['train'] = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )
    dict_transforms['spec_kws_cmd']['val'] = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128,
                n_fft=400,
                win_length=400,
                hop_length=160,
            )
    dict_transforms['spec_emo']['val'] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128,
            n_fft=400,
            win_length=400,
            hop_length=160,
        )
    dict_transforms['text'] = TextTransform()
    return dict_transforms
