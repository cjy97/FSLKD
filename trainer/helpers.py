import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.samplers import CategoriesSampler
from model.base import FSLClassifier

import util.lr_decay as lrd


def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        from dataloader.mini import MiniImageNet as Dataset
    elif args.dataset == 'TieredImageNet':
        from dataloader.tiered import tieredImageNet as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = args.num_workers * num_device if args.multi_gpu else args.num_workers

    trainset = Dataset('train', args, augment=args.augment)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes,
                                      max(args.way, args.num_classes),
                                      args.shot + args.query)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                                    args.num_eval_episodes,
                                    args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)

    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                                     10000,  # args.num_eval_episodes,
                                     args.way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def prepare_model(args):
    model  = FSLClassifier(args)

    total = sum([param.nelement() for param in model.encoder.parameters()])
    print("Parameters of encoder: %.2fM" % (total / 1e6))

    total = sum([param.nelement() for param in model.fsl_head.parameters()])
    print("Parameter of fsl_head: %.2fM" % (total / 1e6))

    # from thop import profile
    # input = torch.randn(80, 3, 84, 84)
    # flops, params = profile(model, inputs=(input,))
    # print("flops: ", flops)
    # print("params: ", params)
    # assert 0

    if args.init_weights is not None:
        model_dict = model.state_dict()

        # if args.backbone == 'ViT':
        #     pretrained_dict = torch.load(args.init_weights, map_location=torch.device('cpu'))['model']
        # else:
        pretrained_dict = torch.load(args.init_weights, map_location=torch.device('cpu'))['params']

        # if args.backbone == 'ConvNet':
        #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        # if args.backbone == 'Vit':
        #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        print("load init_weights: ", pretrained_dict.keys())
        print("model_dict: ", model_dict.keys() )

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load init_weights: ", pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        msg = model.load_state_dict(model_dict)

        print("msg: ", msg)
        if len(msg.missing_keys) != 0:
            print("Missing keys:{}".format(msg.missing_keys), level="warning")
        if len(msg.unexpected_keys) != 0:
            print("Unexpected keys:{}".format(msg.unexpected_keys), level="warning")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # if args.multi_gpu:
    #     model.encoder = nn.DataParallel(model.encoder, dim=0)
    #     para_model = model.to(device)
    # else:
    #     para_model = model.to(device)

    return model #, para_model


def prepare_optimizer(model, args):
    # top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone == 'ConvNet':
        optimizer = optim.Adam(
            # [{'params': model.encoder.parameters()},
            #  {'params': top_para, 'lr': args.lr * args.lr_mul}],
            [{'params': model.encoder.parameters()},
             {'params': model.fsl_head.parameters(), 'lr': args.lr * 1}],

            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )
    elif args.backbone == 'Res12':
        optimizer = optim.SGD(
            # [{'params': model.encoder.parameters()},
            #  {'params': top_para, 'lr': args.lr * args.lr_mul}],
            [{'params': model.encoder.parameters()},
             {'params': model.fsl_head.parameters(), 'lr': args.lr * 1}],

            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )
    elif args.backbone == 'Vit':
        model_without_ddp = model.encoder
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )

        optimizer = optim.AdamW(
            param_groups,
            lr=args.lr
        )
    else:
        raise ValueError('No Such optim')


    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler