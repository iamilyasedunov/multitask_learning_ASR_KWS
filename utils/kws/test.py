from dataset_utils import DatasetDownloader, TrainDataset, SpeechCommandDataset
from sklearn.model_selection import train_test_split
from preprocessing.log_mel_spec import LogMelspec
from model.model import *
from train_utils.utils import *
import warnings
import torch.quantization

warnings.filterwarnings("ignore")

set_seed(21)


def main(config):
    writer = None
    _ = DatasetDownloader(key_word)

    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=config.keyword
    )
    indexes = torch.randperm(len(dataset))
    val_indexes = indexes[int(len(dataset) * 0.8):]

    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    if config['multiply_val_df']:
        val_df = pd.concat([val_df] * config['multiply_val_df']).reset_index()

    val_set = SpeechCommandDataset(csv=val_df)

    print(f"test samples = {len(val_set)}")

    # sampler for oversampling

    # Dataloaders
    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.

    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=2, pin_memory=False)

    loader_for_check_model = DataLoader(val_set, batch_size=1,
                                        shuffle=False, collate_fn=Collator(),
                                        num_workers=2, pin_memory=False)

    melspec_val = LogMelspec(is_train=False, config=config)

    # init model
    trainer = Trainer(writer, config)

    config_writer = {
        "type": "test",
    }
    if config['resume']:
        base_model = CRNN(config)
        print(base_model)

        base_model = base_model.to(config.device)
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint["state_dict"]
        base_model.load_state_dict(state_dict, strict=False)

        print(f"Base model loaded: {config['resume']}")
        get_model_info(base_model, loader_for_check_model, melspec_val, device)
        trainer.validation(base_model, val_loader, melspec_val, config.device, config_writer)

    if config.distillation_soft_labels:
        model = CRNN(config.distillation_soft_labels.student_config)
        print(model)

        model = model.to(config.device)
        checkpoint = torch.load(config.distillation_soft_labels.resume)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Distilled model loaded: {config.distillation_soft_labels.resume}")
        get_model_info(model, loader_for_check_model, melspec_val, device)
        trainer.validation(model, val_loader, melspec_val, config.device, config_writer)

    if config["quantize_dynamic"]:
        # model = model.to('cpu')
        for dict_config in config["quantize_dynamic"]:
            layers_to_quant = dict_config["layers"]
            quant_dtypes = dict_config["dtype"]
            for quant_dtype in quant_dtypes:
                print('*' * 30, "[DYNAMIC QUANT]", '*' * 30)
                print(quant_dtype, layers_to_quant)
                quantized_model = torch.quantization.quantize_dynamic(
                    model, layers_to_quant, dtype=quant_dtype
                )
                trainer.validation(quantized_model, val_loader, melspec_val, config.device, config_writer)
                get_model_info(quantized_model, loader_for_check_model, melspec_val, device)

                # save_torchscript_model(quantized_model, "saved/models/kws_sheila/1110_220958/",
                #                       "linear_gru_int8_dynamic_quant.pt")


if __name__ == "__main__":
    key_word = 'sheila'  # We will use 1 key word -- 'sheila'
    device = torch.device('cpu')  # ('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = {
        'verbosity': 2,
        'name': "test",
        'log_step': 50,
        'exper_name': f"kws_{key_word}_crnn",
        'keyword': key_word,
        'batch_size': 128,
        'len_epoch': 200,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'bidirectional': False,
        'cnn_out_channels': 8,
        'num_epochs': 100,
        'n_mels': 40,  # number of mels for melspectrogram
        'kernel_size': (5, 20),  # size of kernel for convolution layer in CRNN
        'stride': (2, 8),  # size of stride for convolution layer in CRNN
        'hidden_size': 64,  # size of hidden representation in GRU
        'gru_num_layers': 2,  # number of GRU layers in CRNN
        'gru_num_dirs': 2,  # number of directions in GRU (2 if bidirectional)
        'dropout': 0.1,
        'num_classes': 2,  # number of classes (2 for "no word" or "sheila is in audio")
        'sample_rate': 16000,
        'device': device.__str__(),
        'resume': "other/model_acc_2e-05_epoch_42.pth",
        'resume_jit': False,
        'quantize_dynamic': [{"layers": {torch.nn.Linear, nn.GRU}, "dtype": [torch.qint8]}],
                             # [{"layers": {torch.nn.Linear}, "dtype": [torch.float16, torch.qint8, ]},
                             # {"layers": {torch.nn.Linear, nn.GRU}, "dtype": [torch.float16, torch.qint8]}],
        'quantize_static': True,
        'pruning': {"un_structured": False,  # [{"layer": nn.Conv1d, "prob": 0.5},
                    # {"layer": nn.Linear, "prob": 0.5}],
                    "structured": False  # [{"layer": nn.Conv1d, "prob": 0.5},
                    # {"layer": nn.Linear, "prob": 0.5},
                    # {"layer": nn.GRU, "prob": 0.5}]
                    },
        'multiply_val_df': 1,
        'distillation_soft_labels': {
            'mimic_logits': False,
            'soft_labels': False,
            'resume': "other/model_acc_3e-05_epoch_290.pth",
            "student_config": {
                'T': 15.0,
                'lambda_': 0.95,
                'cnn_out_channels': 4,
                'kernel_size': (5, 20),
                'stride': (2, 8),
                'n_mels': 40,
                'hidden_size': 16,
                'gru_num_layers': 1,
                'bidirectional': False,
                'num_classes': 2,
                'dropout': 0.0,
            }

        }
    }

    config = make_config(key_word, config)
    print(f"keyword: '{config.keyword}'\ndevice: {config.device}")
    main(config)
