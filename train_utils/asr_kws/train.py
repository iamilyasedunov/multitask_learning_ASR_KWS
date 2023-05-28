import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.asr.decoder import TextTransform
import numpy as np
from utils.asr.metrics import *
from utils.utils import count_FA_FR
from datetime import datetime

class Trainer:
    def __init__(self, model, device, train_loader, test_loader, optimizer, scheduler, epoch, iter_meter,
                 writer, log_step, config):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.asr_criterion = torch.nn.CTCLoss(blank=28).to(device)
        self.kws_criterion = torch.nn.CrossEntropyLoss()
        self.emo_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.iter_meter = iter_meter
        self.writer = writer
        self.log_step = log_step
        self.config = config
        self.avg_wer = 1.0
        self.avg_au_fa_fr = 1.0
        self.last_test_loss = 1000000
        self.min_val_loss = self.last_test_loss
        self.alpha = self.config['multitask']['alpha_loss']
        self.text_transform = TextTransform()

    def train(self):
        self.model.train()
        data_len = len(self.train_loader.dataset)
        self.epoch += 1
        for batch_idx, _data in tqdm(enumerate(self.train_loader), desc="train", total=len(self.train_loader)):
            self.iter_meter.step()
            self.writer.set_step(self.iter_meter.get(), "train")
            start = datetime.now()
            spec_asr, label_asr, input_length_asr, label_length_asr, spec_kws, label_kws, spec_emo, label_emo = _data
            spec_asr, label_asr = spec_asr.to(self.device), label_asr.to(self.device)
            # print(f"get_data: {(datetime.now() - start).total_seconds()}")

            self.optimizer.zero_grad()

            start = datetime.now()
            output_asr = self.model.forward_asr_head(spec_asr)
            output_asr = F.log_softmax(output_asr, dim=2)
            output_asr = output_asr.transpose(0, 1)  # (time, batch, n_class)
            loss_asr = self.asr_criterion(output_asr, label_asr, input_length_asr, label_length_asr)

            spec_kws, label_kws = spec_kws.to(self.device), label_kws.to(self.device)
            output_kws = self.model.forward_kws_head(spec_kws)
            loss_kws = self.kws_criterion(output_kws, label_kws)

            spec_emo, label_emo = spec_emo.to(self.device), label_emo.to(self.device)
            output_emo = self.model.forward_emo_head(spec_emo)
            loss_emo = self.emo_criterion(output_emo, label_emo)
            # print(f"infer: {(datetime.now() - start).total_seconds()}")

            #_, output_kws = self.model(spec_kws)
            alpha = 1 / 3
            loss = alpha * loss_asr + alpha * loss_kws + alpha * loss_emo
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            if batch_idx % self.log_step == 0 and self.iter_meter.get() > self.log_step or batch_idx == data_len:
                # asr metrics
                decoded_preds, decoded_targets = self.text_transform.greedy_decoder(output_asr.transpose(0, 1),
                                                                                    label_asr, label_length_asr)
                test_cer, test_wer = np.array([]), np.array([])
                for j in range(len(decoded_preds)):
                    test_cer = np.append(test_cer, cer(decoded_targets[j], decoded_preds[j]))
                    test_wer = np.append(test_wer, wer(decoded_targets[j], decoded_preds[j]))

                # kws metrics
                probs_kws = F.softmax(output_kws, dim=-1)
                argmax_probs = torch.argmax(probs_kws, dim=-1)
                FA, FR = count_FA_FR(argmax_probs, label_kws)
                acc_kws = torch.sum(argmax_probs == label_kws).item() / torch.numel(argmax_probs)
                # emo metrics
                probs_emo = F.softmax(output_emo, dim=-1)
                argmax_probs = torch.argmax(probs_emo, dim=-1)
                acc_emo = torch.sum(argmax_probs == label_emo).item() / torch.numel(argmax_probs)
                # print(probs_emo.detach().cpu().numpy())
                # print(argmax_probs.cpu().numpy())
                # print(label_emo.cpu().numpy())
                # print(torch.numel(argmax_probs))

                # log metrics
                self.writer.add_scalars("train_asr/", {'step': self.iter_meter.get(), 'loss': loss_asr.item(),
                                                       'cer': test_cer.mean(), 'wer': test_wer.mean()})

                self.writer.add_scalars("train_kws/", {'loss': loss_kws.item(),
                                                       'acc': acc_kws,
                                                       'FA': FA,
                                                       'FR': FR})

                self.writer.add_scalars("train_emo/", {'loss': loss_emo.item(),
                                        'acc': acc_emo})

                self.writer.add_spectrogram("train_asr/mel", spec_asr[0])
                self.writer.add_spectrogram("train_kws/mel", spec_kws[0])
                #lr = self.scheduler.get_lr()
                self.writer.add_scalars("train/", {'loss': loss.item(),
                                                   #'lr': lr,
                                                   'cnn_body_weight_mean': self.model.cnn.weight.mean().item(),
                                                   'kws_cls_weight_mean': self.model.classifier_kws.weight.mean().item(),
                                                   'asr_cls_weight_mean': self.model.classifier_asr[
                                                       0].weight.mean().item()})

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\twer: {:.6f}\tAcc_kws: {:.6f}\tAcc_emo: {:.6f}'.format(
                #     self.epoch, batch_idx * len(spec_asr), data_len,
                #                 100. * batch_idx / len(self.train_loader), loss.item(), test_wer.mean(), acc_kws, acc_emo)) # lr))
                del probs_kws
                del argmax_probs
            del _data
            del spec_asr
            del label_asr
            del input_length_asr
            del label_length_asr
            del spec_kws, spec_emo
            del label_kws, label_emo
            del output_asr, output_kws, output_emo
            torch.cuda.empty_cache()

    def test(self):

        self.model.eval()
        test_loss = 0
        test_cer, test_wer = np.array([]), np.array([])
        loss_asr_mean, loss_kws_mean, loss_emo_mean, FA_mean, FR_mean, acc_mean_kws, acc_mean_emo = [], [], [], [], [], [], []
        with torch.no_grad():
            for i, _data in tqdm(enumerate(self.test_loader), desc="val", total=len(self.test_loader)):
                spec_asr, label_asr, input_length_asr, label_length_asr, spec_kws, label_kws, spec_emo, label_emo = _data
                spec_asr, label_asr = spec_asr.to(self.device), label_asr.to(self.device)

                output_asr = self.model.forward_asr_head(spec_asr)
                output_asr = F.log_softmax(output_asr, dim=2)
                output_asr = output_asr.transpose(0, 1)  # (time, batch, n_class)
                loss_asr = self.asr_criterion(output_asr, label_asr, input_length_asr, label_length_asr)
                spec_kws, label_kws = spec_kws.to(self.device), label_kws.to(self.device)
                output_kws = self.model.forward_kws_head(spec_kws)
                #_, output_kws = self.model.forward_kws_head(spec_kws)
                loss_kws = self.kws_criterion(output_kws, label_kws)

                spec_emo, label_emo = spec_emo.to(self.device), label_emo.to(self.device)
                output_emo = self.model.forward_emo_head(spec_emo)
                loss_emo = self.emo_criterion(output_emo, label_emo)

                alpha = 1/3
                loss = alpha * loss_asr + alpha * loss_kws + alpha * loss_emo

                loss_asr_mean.append(loss_asr.item())
                loss_kws_mean.append(loss_kws.item())
                loss_emo_mean.append(loss_emo.item())

                test_loss += loss.item() / len(self.test_loader)

                decoded_preds, decoded_targets = self.text_transform.greedy_decoder(output_asr.transpose(0, 1),
                                                                                    label_asr, label_length_asr)
                for j in range(len(decoded_preds)):
                    test_cer = np.append(test_cer, cer(decoded_targets[j], decoded_preds[j]))
                    test_wer = np.append(test_wer, wer(decoded_targets[j], decoded_preds[j]))

                # kws metrics
                probs_kws = F.softmax(output_kws, dim=-1)
                argmax_probs = torch.argmax(probs_kws, dim=-1)
                FA, FR = count_FA_FR(argmax_probs, label_kws)
                acc = torch.sum(argmax_probs == label_kws).item() / torch.numel(argmax_probs)
                FA_mean.append(FA)
                FR_mean.append(FR)
                acc_mean_kws.append(acc)

                # emo metrics
                probs_emo = F.softmax(output_emo, dim=-1)
                argmax_probs = torch.argmax(probs_emo, dim=-1)
                acc_emo = torch.sum(argmax_probs == label_emo).item() / torch.numel(argmax_probs)
                acc_mean_emo.append(acc_emo)
                del _data
                torch.cuda.empty_cache()

        self.last_test_loss = test_loss
        if self.last_test_loss < self.min_val_loss:
            self.min_val_loss = self.last_test_loss
            #self.save_checkpoint(self.epoch)
        self.avg_wer = test_wer.mean()
        self.writer.set_step(self.iter_meter.get(), "val")
        self.writer.add_scalars("val_asr/", {'step': self.iter_meter.get(), 'loss': np.array(loss_asr_mean).mean(),
                                             'cer': test_cer.mean(), 'wer': test_wer.mean()})

        self.writer.add_scalars("val_kws/", {'loss': np.array(loss_kws_mean).mean(),
                                             'acc': np.array(acc_mean_kws).mean(),
                                             'FA': np.array(FA_mean).mean(),
                                             'FR': np.array(FR_mean).mean()})
        self.writer.add_scalars("val_emo/", {'loss': np.array(loss_emo_mean).mean(),
                                             'acc': np.array(acc_mean_emo).mean()})
        self.writer.add_spectrogram("val_asr/mel", spec_asr[0])
        self.writer.add_spectrogram("val_kws/mel", spec_kws[0])
        self.writer.add_spectrogram("val_emo/mel", spec_emo[0])

        self.writer.add_scalars("val/", {'loss': test_loss,
                                         'cnn_body_weight_mean': self.model.cnn.weight.mean().item(),
                                         'kws_cls_weight_mean': self.model.classifier_kws.weight.mean().item(),
                                         'asr_cls_weight_mean': self.model.classifier_asr[0].weight.mean().item()})

        print(
            'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss,
                                                                                              test_cer.mean(),
                                                                                              test_wer.mean()))

    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.config['wandb']['save_dir']}/model_{epoch}_loss_{round(self.last_test_loss, 3)}_" \
                          f"wer_{round(self.avg_wer, 4)}.pth"
        print(f"Saving checkpoint: {checkpoint_path}")
        save_obj = {'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'criterion_asr': self.asr_criterion.state_dict(),
                    'criterion_kws': self.kws_criterion.state_dict(),
                    'criterion_emo': self.emo_criterion.state_dict(),
                    'steps': self.iter_meter.get(),
                    'epoch': epoch}
        torch.save(save_obj, checkpoint_path)
        print(f"Сheckpoint: {checkpoint_path} saved.")


    def load_checkpoint(self):
        checkpoint_obj = torch.load(self.config['multitask']['checkpont_path'])
        self.model.load_state_dict(checkpoint_obj['model'])
        self.optimizer.load_state_dict(checkpoint_obj['optimizer'])
        self.asr_criterion.load_state_dict(checkpoint_obj['criterion_asr'])
        self.kws_criterion.load_state_dict(checkpoint_obj['criterion_kws'])
        self.emo_criterion.load_state_dict(checkpoint_obj['criterion_emo'])
        self.scheduler.load_state_dict(checkpoint_obj['scheduler'])
        self.iter_meter.set(checkpoint_obj['steps'] + 1)
        self.epoch = checkpoint_obj['epoch']
        print(f"Checkpoint loaded: {self.config['multitask']['checkpont_path']}")