import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)

from trainer.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)

import util.lr_sched as lr_sched

from trainer.base_trainer import Trainer

from Distill_module.kd_loss import KD_loss_func

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()

        return label, label_aux

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(args.warmup_epoch + args.max_epoch):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()

            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()

            correct = 0 #
            total = 0   #
            sum_ce_loss = 0.0
            sum_kd_loss = 0.0

            train_len = len(self.train_loader)
            for i, batch in enumerate(self.train_loader):
                self.train_step += 1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                # logits, reg_logits = self.para_model(data)
                # if reg_logits is not None:
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                # else:
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = F.cross_entropy(logits, label)

                results = self.model(data)
                # print("results: ", len(results))
                logits, kd_loss = KD_loss_func(args, results)

                ce_loss = F.cross_entropy(logits, label)
                kd_loss = kd_loss * args.kd_weight
                # print("ce_loss: ", ce_loss)
                # print("kd_loss: ", kd_loss)
                total_loss = ce_loss + kd_loss

                sum_ce_loss += ce_loss
                sum_kd_loss += kd_loss
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == label).sum().item()
                total += label.size()[0]

                tl2.add(ce_loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

            if args.backbone == "Vit":
                lr = lr_sched.adjust_learning_rate(self.optimizer, i / train_len + epoch, args)
            else:
                self.lr_scheduler.step()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            vl, va, vap = self.try_evaluate(epoch)
            self.epoch_record(epoch, lr, vl, va, vap, train_acc=correct / total,
                              avg_ce_loss=sum_ce_loss / len(self.train_loader),
                              avg_kd_loss=sum_kd_loss / len(self.train_loader))

            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, os.path.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))  # loss and acc
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2))  # loss and acc
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                # print("logits: ", logits)
                # print("label: ", label)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                # print("loss: {}; acc: {}".format(loss, acc) )
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval']))

        return vl, va, vap

    def final_record(self):
        # save the best performance in a txt file

        with open(
                os.path.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])),
                'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

    def epoch_record(self, epoch, lr, vl, va, vap, train_acc, avg_ce_loss, avg_kd_loss):
        print(self.args.save_path)
        with open(os.path.join(self.args.save_path, 'record.txt'), 'a') as f:
            f.write(
                'epoch {}: lr={} train_acc={:.4f}, eval_loss={:.4f}, eval_acc={:.4f}+{:.4f}, avg_ce_loss={:.4f}, avg_kd_loss={:.4f}\n'.format(
                    epoch, lr, train_acc, vl, va, vap, avg_ce_loss, avg_kd_loss))