import torch
import torch.nn.functional as F
from utils.asr.decoder import TextTransform
from utils.asr.metrics import *
from tqdm.auto import tqdm


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def set(self, val):
        self.val = val

    def get(self):
        return self.val


class Trainer:
    def __init__(self, model, device, train_loader, test_loader, criterion, optimizer, scheduler, epoch, iter_meter,
                 writer, log_step, config):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.iter_meter = iter_meter
        self.writer = writer
        self.log_step = log_step
        self.config = config
        self.avg_wer = 1.0
        self.last_test_loss = 1000000
        self.min_val_loss = self.last_test_loss
        self.text_transform = TextTransform()

    def train(self):
        self.model.train()
        data_len = len(self.train_loader.dataset)
        self.epoch += 1
        for batch_idx, _data in tqdm(enumerate(self.train_loader), desc="train", total=len(self.train_loader)):
            self.iter_meter.step()
            self.writer.set_step(self.iter_meter.get(), "train")
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = self.criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            if batch_idx % self.log_step == 0 and self.iter_meter.get() > self.log_step or batch_idx == data_len:
                # self.save_checkpoint(self.epoch)
                decoded_preds, decoded_targets = self.text_transform.greedy_decoder(output.transpose(0, 1),
                                                                                    labels, label_lengths)
                test_cer, test_wer = np.array([]), np.array([])
                for j in range(len(decoded_preds)):
                    test_cer = np.append(test_cer, cer(decoded_targets[j], decoded_preds[j]))
                    test_wer = np.append(test_wer, wer(decoded_targets[j], decoded_preds[j]))
                self.writer.add_scalars("train/", {'step': self.iter_meter.get(), 'loss': loss.item(),
                                                   'learning_rate': self.scheduler.get_last_lr(),
                                                   'cer': test_cer.mean(), 'wer': test_wer.mean()})
                self.writer.add_spectrogram("train/mel", spectrograms[0])
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\twer: {:.6f}'.format(
                    self.epoch, batch_idx * len(spectrograms), data_len,
                                100. * batch_idx / len(self.train_loader), loss.item(), test_wer.mean()))
            del _data
            torch.cuda.empty_cache()

    def test(self):
        print('\nevaluating...')

        self.model.eval()
        test_loss = 0
        test_cer, test_wer = [], []
        with torch.no_grad():
            for i, _data in tqdm(enumerate(self.test_loader), desc="val", total=len(self.test_loader)):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)

                output = self.model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)

                loss = self.criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(self.test_loader)

                decoded_preds, decoded_targets = self.text_transform.greedy_decoder(output.transpose(0, 1),
                                                                                    labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
        avg_cer = sum(test_cer) / len(test_cer)
        avg_wer = sum(test_wer) / len(test_wer)
        self.last_test_loss = test_loss
        if self.last_test_loss < self.min_val_loss:
            self.min_val_loss = self.last_test_loss
            self.save_checkpoint(self.epoch)
        self.avg_wer = avg_wer
        self.writer.set_step(self.iter_meter.get(), "val")
        self.writer.add_scalars("val/", {'step': self.iter_meter.get(), 'loss': test_loss,
                                         'cer': avg_cer, 'wer': avg_wer})

        print(
            'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer,
                                                                                              avg_wer))

    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.config['wandb']['save_dir']}/model_{epoch}_loss_{round(self.last_test_loss, 3)}_" \
                          f"wer_{round(self.avg_wer, 4)}.pth"
        print(f"Saving checkpoint: {checkpoint_path}")
        save_obj = {'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'criterion': self.criterion.state_dict(),
                    'steps': self.iter_meter.get(),
                    'epoch': epoch}
        torch.save(save_obj, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_obj = torch.load(self.config['asr']['checkpont_path'])
        self.model.load_state_dict(checkpoint_obj['model'])
        self.optimizer.load_state_dict(checkpoint_obj['optimizer'])
        self.criterion.load_state_dict(checkpoint_obj['criterion'])
        self.iter_meter.set(checkpoint_obj['steps'] + 1)
        self.epoch = checkpoint_obj['epoch']
        print(f"Checkpoint loaded: {self.config['asr']['checkpont_path']}")
