from dataset_utils import SpeechCommandDataset, DatasetDownloader
from model.model import CRNN
from augmentations.augs_creation import AugsCreation
from preprocessing.log_mel_spec import LogMelspec
from train_utils.utils import *
from logger import *
import warnings

warnings.filterwarnings("ignore")

set_seed(21)


def main(config):

    writer = get_writer(config)
    _ = DatasetDownloader(key_word)
    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=config.keyword
    )
    data_len_scaled = int(len(dataset) * config.data_percent) if config.data_percent else len(dataset)
    print(data_len_scaled)
    indexes = torch.randperm(data_len_scaled)
    train_indexes = indexes[:int(data_len_scaled * 0.8)]
    val_indexes = indexes[int(data_len_scaled * 0.8):]

    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set = SpeechCommandDataset(csv=val_df)

    print(f"all train({len(train_set)}) + val samples({len(val_set)}) = {len(train_set) + len(val_set)}")
    student_model = None
    # sampler for oversampling
    train_sampler = get_sampler(train_set.csv['label'].values)

    # Dataloaders
    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=False, collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=2, pin_memory=False)

    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=2, pin_memory=False)

    val_check = DataLoader(val_set, batch_size=1,
                           shuffle=False, collate_fn=Collator(),
                           num_workers=2, pin_memory=False)

    melspec_train = LogMelspec(is_train=True, config=config)
    melspec_val = LogMelspec(is_train=False, config=config)

    # init model
    model = CRNN(config)
    model = model.to(config.device)

    if config.resume:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from checkpoint: {config.resume}")
    print(model)
    get_model_info(model, val_check, melspec_val, config.device)

    if config.distillation_soft_labels.mimic_logits or config.distillation_soft_labels.soft_labels:
        student_model = CRNN(config.distillation_soft_labels.student_config)
        student_model = student_model.to(config.device)
        if config.distillation_soft_labels.resume:
            checkpoint = torch.load(config.distillation_soft_labels.resume)
            state_dict = checkpoint["state_dict"]
            student_model.load_state_dict(state_dict, strict=False)
            print(f"Model STUDENT loaded from checkpoint: {config.distillation_soft_labels.resume}")
        print("STUDENT model:")
        get_model_info(student_model, val_check, melspec_val, config.device)
        opt = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer = Trainer(writer, config)
    min_val_metric = 100
    for n in range(config.num_epochs):
        config_writer = {
            "epoch": n + 1,
            "type": "train",
            "profile_model": False,
        }
        if config.distillation_soft_labels and config.distillation_soft_labels.mimic_logits:
            config_writer["lambda"] = config.distillation_soft_labels.student_config.lambda_
            trainer.train_kd_mimic_logits(model, student_model, opt, train_loader, melspec_train, config.device,
                                          config_writer)
            trainer.validation(student_model, val_loader, melspec_val, config.device, config_writer)

        elif config.distillation_soft_labels and config.distillation_soft_labels.soft_labels:
            config_writer["T"] = config.distillation_soft_labels.student_config.T
            config_writer["lambda"] = config.distillation_soft_labels.student_config.lambda_
            trainer.train_kd_soft_labels(model, student_model, opt, train_loader, melspec_train, config.device,
                                         config_writer)
            trainer.validation(student_model, val_loader, melspec_val, config.device, config_writer)
        else:
            trainer.train_epoch(model, opt, train_loader, melspec_train, config.device, config_writer)

            trainer.validation(model, val_loader, melspec_val, config.device, config_writer)
        val_metric = trainer.get_mean_val_au_fa_fr()
        # print('END OF EPOCH', n)

        if n % 10 == 0 or val_metric < min_val_metric:
            min_val_metric = val_metric if val_metric < min_val_metric else min_val_metric
            model_path = config["save_dir"] + "/" + f"model_acc_{round(val_metric, 5)}_epoch_{n}.pth"

            if config.distillation_soft_labels and student_model is not None:
                state_dict = student_model.state_dict()
            else:
                state_dict = model.state_dict()
            state = {
                "epoch": n,
                "state_dict": state_dict,
                "optimizer": opt.state_dict(),
                "config": config,
            }
            torch.save(state, model_path)
            print(f"Saving checkpoint: {model_path}")


if __name__ == "__main__":
    key_word = 'sheila'  # We will use 1 key word -- 'sheila'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = {
        'data_percent': 1,
        'verbosity': 2,
        'name': "train_upd_crnn",
        'log_step': 10,
        'exper_name': f"kws_{key_word}_crnn_unidirect_teacher",
        'keyword': key_word,
        'batch_size': 128,
        'len_epoch': 1000,
        'learning_rate': 2e-4,
        'weight_decay': 1e-5,
        'num_epochs': 300,
        'bidirectional': False,
        'resume': "other/model_acc_2e-05_epoch_42.pth",
        'cnn_out_channels': 8,
        'n_mels': 40,  # number of mels for melspectrogram
        'kernel_size': (5, 20),  # size of kernel for convolution layer in CRNN
        'stride': (2, 8),  # size of stride for convolution layer in CRNN
        'hidden_size': 64,  # size of hidden representation in GRU
        'gru_num_layers': 2,  # number of GRU layers in CRNN
        'gru_num_dirs': 2,  # number of directions in GRU (2 if bidirectional)
        'num_classes': 2,  # number of classes (2 for "no word" or "sheila is in audio")
        'dropout': 0.1,
        'sample_rate': 16000,
        'device': device.__str__(),
        'structured_pruning_dynamic_quant': False,
        'distillation_soft_labels': {
            'mimic_logits': True,
            'soft_labels': False,
            'resume': False,
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
