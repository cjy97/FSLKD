import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

import os
import numpy as np
from PIL import Image

from dataloader.samplers import CategoriesSampler
from dataloader.mini import MiniImageNet
from dataloader.tiered import tieredImageNet
# from dataloader.cifarFS import

def get_dataloader(args, aug=False):

    num_episodes = args.episodes_per_epoch
    num_workers = args.num_workers

    if args.dataset == 'MiniImageNet':
        trainset = MiniImageNet('train', args, augment=False)
        train_sampler = CategoriesSampler(trainset.label, num_episodes, args.way, args.shot + args.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=True)

        valset = MiniImageNet('val', args)
        val_sampler = CategoriesSampler(valset.label, 200, args.way, args.shot + args.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=True)

        testset = MiniImageNet('test', args)
        test_sampler = CategoriesSampler(testset.label, 1000, args.way, args.shot + args.query)
        test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    elif args.dataset == 'TieredImagenet':
        trainset = tieredImageNet('train', args, augment=False)
        train_sampler = CategoriesSampler(trainset.label, num_episodes, args.way, args.shot + args.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=True)

        valset = tieredImageNet('val', args)
        val_sampler = CategoriesSampler(valset.label, 2000, args.way, args.shot + args.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=True)

        testset = tieredImageNet('test', args)
        test_sampler = CategoriesSampler(testset.label, 10000, args.way, args.shot + args.query)
        test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


    # if dataset == 'cifar10':
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #
    #     train_set = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=True, download=True,
    #                                              transform=transform_train)
    #     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
    #                                                num_workers=num_workers)
    #     test_set = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=False, download=True,
    #                                             transform=transform_test)
    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
    #                                               num_workers=num_workers)
    # elif dataset == 'mnist':
    #     transform_train = transforms.Compose([
    #         transforms.Resize((32, 32)),
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.Resize((32, 32)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    #
    #     train_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=True, download=True,
    #                                            transform=transform_train)
    #     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
    #                                                num_workers=num_workers)
    #     test_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=False, download=True,
    #                                           transform=transform_test)
    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
    #                                               num_workers=num_workers)
    #
    # elif dataset == 'ocean_human' or dataset == 'ocean_ship':
    #     flag = dataset.split("ocean_")[1]
    #
    #     dataset_train = Ocean_data("dataset/ocean/train", "dataset/ocean/label.csv", flag)
    #     train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
    #                                   num_workers=num_workers)
    #
    #     dataset_test = Ocean_data("dataset/ocean/test", "dataset/ocean/label.csv", flag)
    #     test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=True,
    #                                                num_workers=num_workers)
    #
    # else:
    #     raise Exception('Not dataset:{0}'.format(dataset))
    #
    # return train_loader, test_loader



# class Ocean_data(torch.utils.data.Dataset):
#     def __init__(self, data_dir, label_dir, flag):
#         super().__init__()
#         self.name_list = os.listdir(data_dir)  # 获得子目录下的图片的名称
#         self.label = np.loadtxt(label_dir, delimiter=',')
#         self.imgpath = data_dir
#         self.flag = flag
#         self.transform = transforms.Compose(
#             [
#                 # transforms.Resize(size = (512,512)),#尺寸规范
#                 # transforms.RandomResizedCrop((512,512)),
#                 # transforms.RandomCrop((512,512), padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 # transforms.RandomRotation(45),
#                 # transforms.ColorJitter(contrast=0.5),
#                 transforms.Grayscale(num_output_channels=1),
#                 transforms.ToTensor(),  # 转化为tensor
#                 # transforms.Normalize((0.5), (0.5)),
#
#             ])  # Transforms只适用于PIL 中的Image打开的图像
#
#     def __getitem__(self, index):
#         name = self.name_list[index]  # 获得当前图片的名称
#         num = int(name[:-4])
#         path = os.path.join(self.imgpath, name)
#         image = Image.open(path)
#         # image = np.expand_dims(image,axis=0)
#         # image = torch.FloatTensor(image).permute(2,0,1)
#         image = self.transform(image)
#         if self.flag == 'human':
#             if self.label[num][1] == 1:
#                 label = 1
#             else:
#                 label = 0
#         elif self.flag == 'ship':
#             if self.label[num][2] == 1:
#                 label = 1
#             else:
#                 label = 0
#         else:
#             if self.label[num][2] == 1 or self.label[num][1] == 1:
#                 label = 1
#             else:
#                 label = 0
#         # label =  np.reshape(label,(1,))
#         label = torch.as_tensor(label, dtype=torch.int64)
#         # label = torch.FloatTensor(label)
#         return image, label
#
#     def __len__(self):
#         return len(self.name_list)