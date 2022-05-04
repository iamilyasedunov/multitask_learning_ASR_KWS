import sys
from utils.utils import *
from datetime import datetime

sys.path.append(sys.path[0] + "/..")


class Trainer():
    def __init__(self, writer, config):
        if "test" not in config["name"] and writer is None:
            print(f"Error: init writer for train section!")
            raise ValueError
        self.writer = writer
        self.config = config
        self.step = 0
        self.train_metrics = {
            "train_losses": [], "train_accs": [], "train_FAs": [], "train_FRs": [],
        }
        self.val_metrics = {
            "val_losses": [], "val_accs": [], "val_FAs": [], "val_FRs": [], "val_au_fa_fr": [], 'val_time_inference': []
        }
        self.last_val_metric = 0

    def get_mean_val_au_fa_fr(self):
        return self.last_val_metric

    def log_train(self, logits, loss, labels):
        probs = F.softmax(logits, dim=-1)
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.sum(argmax_probs == labels).item() / torch.numel(argmax_probs)

        self.train_metrics["train_losses"].append(loss)
        self.train_metrics["train_accs"].append(acc)
        self.train_metrics["train_FAs"].append(FA)
        self.train_metrics["train_FRs"].append(FR)

        if self.step % self.config["log_step"] == 0:
            if self.writer is not None:
                self.writer.set_step(self.step)
                self.writer.add_scalars("train", {'loss': np.mean(self.train_metrics["train_losses"]),
                                                  'acc': np.mean(self.train_metrics["train_accs"]),
                                                  'FA': np.mean(self.train_metrics["train_FAs"]),
                                                  'FR': np.mean(self.train_metrics["train_FRs"])})
            self.train_metrics = {
                "train_losses": [], "train_accs": [], "train_FAs": [], "train_FRs": [],
            }

    def log_after_train_epoch(self, config_writer):
        if self.writer is not None:
            self.writer.add_scalar(f"epoch", config_writer["epoch"])
        else:
            print({'loss': np.mean(self.train_metrics["train_losses"]),
                   'acc': np.mean(self.train_metrics["train_accs"]),
                   'FA': np.mean(self.train_metrics["train_FAs"]),
                   'FR': np.mean(self.train_metrics["train_FRs"])})
        print(f"Epoch end, acc {np.mean(self.train_metrics['train_accs'])}")

    def train_kd_mimic_logits(self, teacher, student, opt, loader, log_melspec, device, config_writer):
        teacher.eval()
        student.train()
        for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
            self.step += 1
            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch)

            opt.zero_grad()

            logits_st = student(batch)
            with torch.no_grad():
                logits_teach = teacher(batch)

            loss = F.mse_loss(logits_st, logits_teach)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)

            opt.step()

            # logging
            self.log_train(logits_st, loss.item(), labels)

        self.log_after_train_epoch(config_writer)

    def train_kd_soft_labels(self, teacher, student, opt, loader, log_melspec, device, config_writer):
        def softXEnt(input_, target_):
            logprobs = F.log_softmax(input_, dim=-1)
            return -(target_ * logprobs).sum() / input_.shape[0]
        teacher.eval()
        student.train()
        T = config_writer['T']
        for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
            self.step += 1
            batch, labels = batch.to(device), labels.to(device)
            batch_prepared = log_melspec(batch)

            logits_st = student(batch_prepared)
            with torch.no_grad():
                logits_teach = teacher(batch_prepared) #.detach()

            hard_predictions = F.softmax(logits_st, dim=-1)
            soft_predictions = F.softmax(logits_st / T, dim=-1)
            soft_labels = F.softmax(logits_teach / T, dim=-1)
            distillation_loss = softXEnt(soft_predictions, soft_labels) / (T ** 2)
            student_loss = F.cross_entropy(hard_predictions, labels)

            loss = config_writer['lambda'] * distillation_loss + (1.0 - config_writer['lambda']) * student_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            opt.step()

            opt.zero_grad()

            # logging
            self.log_train(logits_st, loss.item(), labels)

        self.log_after_train_epoch(config_writer)

    def train_epoch(self, model, opt, loader, log_melspec, device, config_writer):
        model.train()
        acc = torch.tensor([0.0])

        for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
            self.step += 1

            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch)

            opt.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()

            # logging
            self.log_train(logits, loss.item(), labels)

        self.log_after_train_epoch(config_writer)

    @torch.no_grad()
    def validation(self, model, loader, log_melspec, device, config_writer):
        model.eval()
        self.val_metrics = {
            "val_losses": [], "val_accs": [], "val_FAs": [], "val_FRs": [], "val_au_fa_fr": [], 'val_time_inference': []
        }
        all_probs, all_labels = [], []
        for i, (batch, labels) in tqdm(enumerate(loader), desc="val", total=len(loader)):
            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch)

            start = datetime.now()

            output = model(batch)
            probs = F.softmax(output, dim=-1)  # we need probabilities so we use softmax & CE separately

            if config_writer["type"] == "train":
                loss = F.cross_entropy(output, labels)

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            time_infer = (datetime.now() - start).total_seconds()
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(labels.cpu())
            acc = torch.sum(argmax_probs == labels).item() / torch.numel(argmax_probs)
            FA, FR = count_FA_FR(argmax_probs, labels)

            if config_writer["type"] == "train":
                self.val_metrics["val_losses"].append(loss.item())
            else:
                self.val_metrics["val_losses"].append(0)

            self.val_metrics["val_accs"].append(acc)
            self.val_metrics["val_FAs"].append(FA)
            self.val_metrics["val_FRs"].append(FR)
            self.val_metrics["val_time_inference"].append(time_infer)
        # area under FA/FR curve for whole loader
        au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)

        print({'mean_val_loss': round(np.mean(self.val_metrics["val_losses"]), 5),
               'mean_val_acc': round(np.mean(self.val_metrics["val_accs"]), 5),
               'mean_val_FA': round(np.mean(self.val_metrics["val_FAs"]), 5),
               'mean_val_FR': round(np.mean(self.val_metrics["val_FRs"]), 5),
               'val_time_inference': round(np.mean(self.val_metrics["val_time_inference"]), 5),
               'au_fa_fr': round(au_fa_fr, 5)})
        self.last_val_metric = au_fa_fr
        self.step += 1
        if config_writer["type"] == "train" and self.writer is not None:
            self.writer.set_step(self.step, "valid")
            self.writer.add_scalars("val", {'mean_loss': np.mean(self.val_metrics["val_losses"]),
                                            'mean_acc': np.mean(self.val_metrics["val_accs"]),
                                            'mean_FA': np.mean(self.val_metrics["val_FAs"]),
                                            'mean_FR': np.mean(self.val_metrics["val_FRs"]),
                                            'au_fa_fr': au_fa_fr})

        return np.mean(self.val_metrics["val_losses"])
