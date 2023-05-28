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
        self.cmd_criterion = torch.nn.CrossEntropyLoss()
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
            spec_asr, label_asr, input_length_asr, label_length_asr, spec_kws_cmd, label_cmd, label_kws = _data
            spec_asr, label_asr = spec_asr.to(self.device), label_asr.to(self.device)
            # print(f"get_data: {(datetime.now() - start).total_seconds()}")

            self.optimizer.zero_grad()

            start = datetime.now()
            output_asr = self.model.forward_asr_head(spec_asr)
            output_asr = F.log_softmax(output_asr, dim=2)
            output_asr = output_asr.transpose(0, 1)  # (time, batch, n_class)
            loss_asr = self.asr_criterion(output_asr, label_asr, input_length_asr, label_length_asr)

            spec_kws_cmd, label_cmd, label_kws = spec_kws_cmd.to(self.device), label_cmd.to(self.device), label_kws.to(self.device)
            output_kws = self.model.forward_kws_head(spec_kws_cmd)
            loss_kws = self.kws_criterion(output_kws, label_kws)

            output_cmd = self.model.forward_cmd_head(spec_kws_cmd)
            loss_cmd = self.cmd_criterion(output_cmd, label_cmd)
            # print(f"infer: {(datetime.now() - start).total_seconds()}")

            #_, output_kws = self.model(spec_kws_cmd)
            alpha = 1 / 3
            loss = alpha * loss_asr + alpha * loss_kws + alpha * loss_cmd
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
                # cmd metrics
                probs_cmd = F.softmax(output_cmd, dim=-1)
                argmax_probs = torch.argmax(probs_cmd, dim=-1)
                acc_cmd = torch.sum(argmax_probs == label_cmd).item() / torch.numel(argmax_probs)
                # print(probs_cmd.detach().cpu().numpy())
                # print(argmax_probs.cpu().numpy())
                # print(label_cmd.cpu().numpy())
                # print(torch.numel(argmax_probs))

                # log metrics
                self.writer.add_scalars("train_asr/", {'step': self.iter_meter.get(), 'loss': loss_asr.item(),
                                                       'cer': test_cer.mean(), 'wer': test_wer.mean()})

                self.writer.add_scalars("train_kws/", {'loss': loss_kws.item(),
                                                       'acc': acc_kws,
                                                       'FA': FA,
                                                       'FR': FR})

                self.writer.add_scalars("train_cmd/", {'loss': loss_cmd.item(),
                                        'acc': acc_cmd})

                self.writer.add_spectrogram("train_asr/mel", spec_asr[0])
                self.writer.add_spectrogram("train_kws/mel", spec_kws_cmd[0])
                #lr = self.scheduler.get_lr()
                self.writer.add_scalars("train/", {'loss': loss.item(),
                                                   #'lr': lr,
                                                   'cnn_body_weight_mean': self.model.cnn.weight.mean().item(),
                                                   'kws_cls_weight_mean': self.model.classifier_kws.weight.mean().item(),
                                                   'asr_cls_weight_mean': self.model.classifier_asr[
                                                       0].weight.mean().item()})

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\twer: {:.6f}\tAcc_kws: {:.6f}\tAcc_cmd: {:.6f}'.format(
                #     self.epoch, batch_idx * len(spec_asr), data_len,
                #                 100. * batch_idx / len(self.train_loader), loss.item(), test_wer.mean(), acc_kws, acc_cmd)) # lr))
                del probs_kws
                del argmax_probs
            del _data
            del spec_asr
            del label_asr
            del input_length_asr
            del label_length_asr
            del spec_kws_cmd
            del label_kws, label_cmd
            del output_asr, output_kws, output_cmd
            torch.cuda.empty_cache()

    def test(self):

        self.model.eval()
        test_loss = 0
        test_cer, test_wer = np.array([]), np.array([])
        loss_asr_mean, loss_kws_mean, loss_cmd_mean, FA_mean, FR_mean, acc_mean_kws, acc_mean_cmd = [], [], [], [], [], [], []
        with torch.no_grad():
            for i, _data in tqdm(enumerate(self.test_loader), desc="val", total=len(self.test_loader)):
                spec_asr, label_asr, input_length_asr, label_length_asr, spec_kws_cmd, label_cmd, label_kws = _data
                spec_asr, label_asr = spec_asr.to(self.device), label_asr.to(self.device)

                output_asr = self.model.forward_asr_head(spec_asr)
                output_asr = F.log_softmax(output_asr, dim=2)
                output_asr = output_asr.transpose(0, 1)  # (time, batch, n_class)
                loss_asr = self.asr_criterion(output_asr, label_asr, input_length_asr, label_length_asr)
                spec_kws_cmd, label_kws, label_cmd  = spec_kws_cmd.to(self.device), label_kws.to(self.device), label_cmd.to(self.device)
                output_kws = self.model.forward_kws_head(spec_kws_cmd)
                #_, output_kws = self.model.forward_kws_head(spec_kws_cmd)
                loss_kws = self.kws_criterion(output_kws, label_kws)

                output_cmd = self.model.forward_cmd_head(spec_kws_cmd)
                loss_cmd = self.cmd_criterion(output_cmd, label_cmd)

                alpha = 1/3
                loss = alpha * loss_asr + alpha * loss_kws + alpha * loss_cmd

                loss_asr_mean.append(loss_asr.item())
                loss_kws_mean.append(loss_kws.item())
                loss_cmd_mean.append(loss_cmd.item())

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

                # cmd metrics
                probs_cmd = F.softmax(output_cmd, dim=-1)
                argmax_probs = torch.argmax(probs_cmd, dim=-1)
                acc_cmd = torch.sum(argmax_probs == label_cmd).item() / torch.numel(argmax_probs)
                acc_mean_cmd.append(acc_cmd)
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
        self.writer.add_scalars("val_cmd/", {'loss': np.array(loss_cmd_mean).mean(),
                                             'acc': np.array(acc_mean_cmd).mean()})
        self.writer.add_spectrogram("val_asr/mel", spec_asr[0])
        self.writer.add_spectrogram("val_kws_cmd/mel", spec_kws_cmd[0])

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
                    'criterion_cmd': self.cmd_criterion.state_dict(),
                    'steps': self.iter_meter.get(),
                    'epoch': epoch}
        torch.save(save_obj, checkpoint_path)
        print(f"Сheckpoint: {checkpoint_path} saved.")


    def load_checkpoint(self, strict=False):
        checkpoint_obj = torch.load(self.config['multitask']['checkpont_path'])
        chkpt_model = checkpoint_obj['model']
        if strict:
            for key in list(chkpt_model.keys()):
                print(key)
                for tag in ['cmd', 'emo', 'kws']:
                    if tag in key:
                        print(tag, key)
                        # new_key = key.replace('emo', 'cmd')
                        chkpt_model[key] = self.model.state_dict()[key]
                        # del chkpt_model[key]
            for cmd_tag in ['classifier_cmd.weight', 'classifier_cmd.bias']:
                if cmd_tag not in list(chkpt_model.keys()):
                    chkpt_model[cmd_tag] = self.model.state_dict()[cmd_tag]

        self.model.load_state_dict(chkpt_model)
        if not strict:
            self.optimizer.load_state_dict(checkpoint_obj['optimizer'])
        self.asr_criterion.load_state_dict(checkpoint_obj['criterion_asr'])
        self.kws_criterion.load_state_dict(checkpoint_obj['criterion_kws'])
        if not strict:
            self.cmd_criterion.load_state_dict(checkpoint_obj['criterion_cmd'])
        self.scheduler.load_state_dict(checkpoint_obj['scheduler'])
        self.iter_meter.set(checkpoint_obj['steps'] + 1)
        self.epoch = checkpoint_obj['epoch']
        print(f"Checkpoint loaded: {self.config['multitask']['checkpont_path']}")