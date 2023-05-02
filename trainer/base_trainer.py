import os
import abc
import json
import torch

from collections import defaultdict, OrderedDict
from tensorboardX import SummaryWriter

from trainer.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.logger_path = os.path.join(log_dir, 'scalars.json')
        self.tb_logger = SummaryWriter(
                            logdir=os.path.join(log_dir, 'tflogger'),
                            **kwargs,
                        )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict)

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = os.path.join(os.path.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['model/models', 'model/networks', __file__],
        # )
        self.logger = Logger(args, os.path.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass

    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass

    @abc.abstractmethod
    def final_record(self):
        pass

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            vl, va, vap = self.evaluate(self.val_loader)
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va), self.train_epoch)
            print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

        return vl, va, vap  ###

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc', ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm', tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, ' \
                  'forward_timer: {:.2f} sec,' \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                self.dt.item(), self.ft.item(),
                self.bt.item(), self.ot.item())
            )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            os.path.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
