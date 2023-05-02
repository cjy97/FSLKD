import pprint
import numpy as np
import torch
import os
import random
import argparse
import time


_utils_pp = pprint.PrettyPrinter()
def _pprint(x):
    _utils_pp.pprint(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet', 'CUB'])
    parser.add_argument('--dataset_path', type=str, default='D:/miniImageNet--ravi')
    parser.add_argument('--backbone', type=str, default='Res12', choices=['Res12', 'Vit'])
    parser.add_argument('--classifier', type=str, default='DN4', choices=['ProtoNet', 'DN4', 'RelationNet'])

    parser.add_argument('--episodes_per_epoch', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--temperature', type=int, default=1)

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)

    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)

    parser.add_argument('--init_weights', type=str, default='MiniImageNet-Res12-DN4.pth')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--val_interval', type=int, default=1)


    parser.add_argument("--is_distill", action="store_true", default=False)
    parser.add_argument('--is_prune', action="store_true", default=False)

    return parser

def postprocess_args(args):
    save_path1 = '-'.join(
        [args.dataset, args.backbone, args.classifier, '{}w{}s'.format(args.way, args.shot)]
    )
    if args.lr_scheduler == "step" or args.lr_scheduler == "multistep":
        lr_scheduler_info = args.lr_scheduler + '-step{}'.format(args.step_size) + '-gamma{}'.format(args.gamma)
    else:
        lr_scheduler_info = args.lr_scheduler

    save_path2 = '_'.join(
        ['lr{:.2g}'.format(args.lr), 'weight_decay{}'.format(args.weight_decay), args.optim, lr_scheduler_info, \
         'epoch{}'.format(args.max_epoch), 'temp{}'.format(args.temperature), str(time.strftime('%Y%m%d_%H%M%S'))])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    if not os.path.exists(os.path.join(args.save_dir, save_path1, save_path2)):
        os.mkdir(os.path.join(args.save_dir, save_path1, save_path2))

    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)

    f = open(os.path.join(args.save_path, "record.txt"), 'w')
    f.close()

    return args


def save_checkpoint(args, filename, model, optimizer, lr_scheduler, epoch):
    state = {
        'args': args,
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(state, os.path.join(args.save_path, filename+'.pth'))
