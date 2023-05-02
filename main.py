# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import random
import argparse

import torch
from dataloader.datasets import get_dataloader

# 主文件
from Quant_module import *
from model.backbone import *
from model.base import FSLClassifier

from util.utils import count_acc, compute_confidence_interval, set_random_seed, get_command_line_parser, postprocess_args, save_checkpoint


def train(args, model, device, optimizer, lr_scheduler, train_loader, val_loader, init_epoch):
    train_len = len(train_loader)
    record = np.zeros((train_len, 2))

    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    max_acc = 0.0   # best eval acc

    for epoch in range(init_epoch, args.max_epoch):
        model.train()

        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()

        for i, batch in enumerate(train_loader):
            samples, _ = batch  # gt_label is of no use
            # samples, label = batch
            samples = samples.to(device)

            logits = model(samples)
            # print("logits: ", logits)
            # print("label: ", label)
            loss = criterion(logits, label)
            acc = count_acc(logits, label)

            print("loss:{}, acc:{}".format(loss.item(), acc))
            record[i, 0] = loss.item()
            record[i, 1] = acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        tl, _ = compute_confidence_interval(record[:, 0])
        ta, _ = compute_confidence_interval(record[:, 1])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("epoch {}  current learning rate: {}; train_loss={:.4f}, train_acc={:.4f}".format(epoch, lr, tl, ta))

        with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
            f.write("epoch {}, train, lr:{}, train_loss={:.4f}, train_acc={:.4f}\n".format(epoch, lr, tl, ta))

        # evaluate
        if epoch % args.val_interval == 0:
            model.eval()

            val_len = len(val_loader)
            val_record = np.zeros((val_len, 2))

            label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
            label = label.type(torch.LongTensor)
            if torch.cuda.is_available():
                label = label.cuda()

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data, _ = batch

                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    val_record[i, 0] = loss.item()
                    val_record[i, 1] = acc

            vl, _ = compute_confidence_interval(val_record[:, 0])
            va, _ = compute_confidence_interval(val_record[:, 1])
            print("epoch {}, val, eval_loss_={:.4f} eval_acc={:.4f} ".format(epoch, vl, va))
            print(args.save_path)
            with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
                f.write("epoch {}, val, eval_loss={:.4f} eval_acc={:.4f}\n".format(epoch, vl, va))

            if va > max_acc:
                max_acc = va
                with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
                    f.write("Epoch {} : save max_acc  {} !\n".format(epoch, max_acc))
                save_checkpoint(args, 'max_acc', model, optimizer, lr_scheduler, epoch)

        save_checkpoint(args, 'epoch-last', model, optimizer, lr_scheduler, epoch)

def final_test(args, model, test_loader):
    if os.path.exists(os.path.join(args.save_path, 'max_acc.pth')):
        msg = model.load_state_dict(torch.load(os.path.join(args.save_path, 'max_acc.pth'))['state_dict'])
        print(msg)

    model.eval()

    test_len = len(test_loader)
    test_record = np.zeros((test_len, 1))

    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data, _ = batch

            logits = model(data)
            # loss = F.cross_entropy(logits, label)
            print("logits: ", logits.size())
            print("label: ", label.size())
            acc = count_acc(logits, label)
            print(" acc: ", acc)
            test_record[i, 0] = acc

    ta, _ = compute_confidence_interval(test_record[:, 0])
    print(" test, acc={:.4f} ".format(ta))
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write(" test, acc={:.4f}\n".format(ta))

    return ta


def main(args):

    # 量化测试程序需要指定的三个参数：
    # model_type = "mlp"  # 以./backbone/resnet_simple.py下定义的resnet20网络为例
    # classifer_type = ""
    #
    # dataset_type = "cifar10"  # 使用的数据集
    # pre_path = "./checkpoints/mlp.pth"  # 模型预训练权重路径

    # prepare dataloader

    # num_episodes = args.episodes_per_epoch
    # num_workers  = args.num_workers
    #
    # trainset = Dataset('train', args, augment=False)
    # train_sampler = CategoriesSampler(trainset.label, num_episodes, args.way, args.shot + args.query)
    # train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    #
    # valset = Dataset('val', args)
    # val_sampler = CategoriesSampler(valset.label, 2000, args.way, args.shot + args.query)
    # val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    #
    # testset = Dataset('test', args)
    # test_sampler = CategoriesSampler(testset.label, 10000, args.way, args.shot + args.query)
    # test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    train_loader, val_loader, test_loader = get_dataloader(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # model = classifier.__dict__[args.classifier](args)
    model  = FSLClassifier(args)
    print("model: ", model)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.to(device)

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay)
    elif args.optim == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError('No Such optim')

    if args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        args.max_epoch,
                        eta_min=0
                    )
    else:
        raise ValueError('No Such lr_scheduler')

    if args.resume is not None and args.init_weights is not None:
        assert 0

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)

        init_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])
        pretrained_dict = state['state_dict']
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}

        msg = model.load_state_dict(pretrained_dict)
        print(msg)
    else:
        init_epoch = 0

    if args.init_weights is not None:
        pretrained_dict = torch.load("Res12-pre.pth", map_location=device)['params']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        for k, _ in model_dict.items():
            if k not in pretrained_dict:
                print("Dismatch weights!")
                assert 0

        msg =  model.load_state_dict(model_dict)
        print(msg)

    # train(args, model, device, optimizer, lr_scheduler, train_loader, val_loader, init_epoch)


    # quant_Model = PTSQ_process(args, model, optimizer, lr_scheduler, train_loader)
    quant_Model = QAT_process(args, model, optimizer, lr_scheduler, train_loader)
    correct_rate = final_test(args, quant_Model, test_loader)
    print("correct_rate: ", correct_rate)



    # Quan_test(model_type, classifer_type, dataset_type, pre_path=pre_path)


    # pass


def Quan_test(model_type, classifer_type, dataset_type, pre_path):
    # 通过接口使用提供的代码，进行测试

    # quant_model = PTDQ(model_type, pre_path, dataset_type)   # 训练后动态量化
    quant_model = PTSQ(model_type, classifer_type, pre_path, dataset_type)   # 训练后静态量化
    # quant_model = QAT(model_type, pre_path, dataset_type)  # 量化感知训练

    torch.save(quant_model.state_dict(), "quantized_model.pth")




def My_quant_matmul(a, b, Int_Type = torch.int32):  # 自定义量化矩阵乘法：接收两个quint8类型的量化张量，返回矩阵乘法后的结果（同样量化为qint8类型）
    assert a.dtype == b.dtype

    if a.dtype == torch.float:
        c = torch.matmul(a, b)   # torch.float32
        return c


    aint = a.int_repr() # torch.uint8
    bint = b.int_repr()

    s1 = str(a)
    a_scale = float(s1.split("scale=")[1].split(",")[0])
    a_zp = int(s1.split('zero_point=')[1].split(')')[0])

    s2 = str(b)
    b_scale = float(s2.split("scale=")[1].split(",")[0])
    b_zp = int(s2.split('zero_point=')[1].split(')')[0])

    cint = torch.matmul(aint.type(Int_Type) - a_zp, bint.type(Int_Type) - b_zp)
    # print("cint: ", cint.dtype)   # 结果类型同 Int_Type

    c = cint * a_scale * b_scale
    print("c: ", c.dtype)   # torch.float32
    print(c)

    scale_c = float((torch.max(c) - torch.min(c)) / (255.0 - 0.0))
    zp_c = int(255 - torch.max(c) / scale_c)
    c = torch.quantize_per_tensor(c, scale=scale_c, zero_point=zp_c, dtype=torch.quint8)

    return c

if __name__ == '__main__':

    # parser = get_command_line_parser()
    # args = postprocess_args(parser.parse_args())
    #
    # set_random_seed(seed=1)
    #
    # main(args)

    a = torch.randn(1, 4, 3)
    b = torch.randn(1, 3, 4)
    # print(a.dtype, b.dtype)

    c = My_quant_matmul(a, b)
    print("c: ", c.dtype)
    print(c)


    scale_a = float((torch.max(a) - torch.min(a)) / (255.0-0.0))
    zp_a = int(255 - torch.max(a)/scale_a)

    scale_b = float((torch.max(b) - torch.min(b)) / (255.0 - 0.0))
    zp_b = int(255 - torch.max(b) / scale_b)


    a = torch.quantize_per_tensor(a, scale=scale_a, zero_point=zp_a, dtype=torch.quint8)
    b = torch.quantize_per_tensor(b, scale=scale_b, zero_point=zp_b, dtype=torch.quint8)
    # print(a.dtype, b.dtype)

    c = My_quant_matmul(a, b, Int_Type=torch.int64)
    print("c: ", c.dtype)
    print(c)