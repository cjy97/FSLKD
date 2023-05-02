import numpy as np
import torch
import torch.nn as nn


def prune_resnet(model, remove_ratio):
    print("Pruning: resnet12")

    ckpt = model.state_dict()

    channels = [[3, 64], [64, 160], [160, 320], [320, 640]]

    conv_weights = []
    bn_weights, bn_biass, bn_running_means, bn_running_vars = [], [], [], []
    ds_weights = []
    ds_bn_weights, ds_bn_biass, ds_bn_running_means, ds_bn_running_vars = [], [], [], []

    ds_mask_nums = []  # 记录ds剪枝对应的masks下标

    for k, v in ckpt.items():
        if 'conv' in k:
            conv_weights.append(v)
        elif ('bn1.weight' in k) or ('bn2.weight' in k) or ('bn3.weight' in k):
            bn_weights.append(v)
        elif ('bn1.bias' in k) or ('bn2.bias' in k) or ('bn3.bias' in k):
            bn_biass.append(v)
        elif ('bn1.running_mean' in k) or ('bn2.running_mean' in k) or ('bn3.running_mean' in k):
            bn_running_means.append(v)
        elif ('bn1.running_var' in k) or ('bn2.running_var' in k) or ('bn3.running_var' in k):
            bn_running_vars.append(v)
        elif 'downsample.0.weight' in k:
            num = len(bn_weights)
            # ds_mask_nums.append([num, num+2])
            ds_mask_nums.append([num - 3, num])
            ds_weights.append(v)
        elif 'downsample.1.weight' in k:
            ds_bn_weights.append(v)
        elif 'downsample.1.bias' in k:
            ds_bn_biass.append(v)
        elif 'downsample.1.running_mean' in k:
            ds_bn_running_means.append(v)
        elif 'downsample.1.running_var' in k:
            ds_bn_running_vars.append(v)

    # 每个卷积层的参数（四维张量）关于输出通道求均值，作为重要性评分
    weight_scores = [(torch.mean(torch.abs(conv_weight), axis=[0, 2, 3])) for conv_weight in
                     conv_weights[1:]]  # 是否跳过第一个卷积层？

    masks = [torch.tensor([True for _ in range(3)])]
    for weight_score in weight_scores:
        weight_score_sort, _ = torch.sort(weight_score)
        remove_num = int(remove_ratio * len(weight_score))
        threshold = weight_score_sort[remove_num]
        mask = weight_score.ge(threshold)
        masks.append(mask)
    masks.append(torch.tensor([True for _ in range(channels[-1][-1])]))

    pruning_conv_weights = []
    pruning_bn_weights = []
    pruning_bn_biass = []
    pruning_bn_running_means = []
    pruning_bn_running_vars = []

    c_in_out = []

    for i in range(len(conv_weights)):
        mask1, mask2 = masks[i], masks[i + 1]
        mask1 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy()))).tolist()
        mask2 = np.squeeze(np.argwhere(np.asarray(mask2.cpu().numpy()))).tolist()

        # 对卷积层剪枝
        conv_weight = conv_weights[i]
        # print('剪枝前尺寸: ', conv_weight.shape)
        conv_weight = conv_weight[:, mask1, :, :]
        conv_weight = conv_weight[mask2, :, :, :]
        # print('剪枝后尺寸: ', conv_weight.shape)
        pruning_conv_weights.append(conv_weight)

        # 对bn层进行剪枝
        bn_weight = bn_weights[i]
        bn_weight = bn_weight[mask2]
        pruning_bn_weights.append(bn_weight)

        bn_bias = bn_biass[i]
        bn_bias = bn_bias[mask2]
        pruning_bn_biass.append(bn_bias)

        bn_running_var = bn_running_vars[i]
        bn_running_var = bn_running_var[mask2]
        pruning_bn_running_vars.append(bn_running_var)

        bn_running_mean = bn_running_means[i]
        bn_running_mean = bn_running_mean[mask2]
        pruning_bn_running_means.append(bn_running_mean)

        # print('[i = %d in = %d out = %d]' % (i+1, len(mask1), len(mask2)))
        c_in_out.append([len(mask1), len(mask2)])

    # 将普通卷积层剪枝后的参数保存到模型文件
    num1, num2, num3, num4, num5 = 0, 0, 0, 0, 0
    for k, v in ckpt.items():
        if 'conv' in k:
            ckpt[k] = pruning_conv_weights[num1]
            num1 = num1 + 1
        elif ('bn1.weight' in k) or ('bn2.weight' in k) or ('bn3.weight' in k):
            ckpt[k] = pruning_bn_weights[num2]
            num2 = num2 + 1
        elif ('bn1.bias' in k) or ('bn2.bias' in k) or ('bn3.bias' in k):
            ckpt[k] = pruning_bn_biass[num3]
            num3 = num3 + 1
        elif ('bn1.running_mean' in k) or ('bn2.running_mean' in k) or ('bn3.running_mean' in k):
            ckpt[k] = pruning_bn_running_means[num4]
            num4 = num4 + 1
        elif ('bn1.running_var' in k) or ('bn2.running_var' in k) or ('bn3.running_var' in k):
            ckpt[k] = pruning_bn_running_vars[num5]
            num5 = num5 + 1

    # 对ds卷积层、bn层进行剪枝
    pruning_ds_weights = []
    pruning_ds_bn_weights = []
    pruning_ds_bn_biass = []
    pruning_ds_bn_running_means = []
    pruning_ds_bn_running_vars = []
    for i in range(len(ds_weights)):
        mask1 = masks[ds_mask_nums[i][0]]
        mask2 = masks[ds_mask_nums[i][1]]
        mask1 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy()))).tolist()
        mask2 = np.squeeze(np.argwhere(np.asarray(mask2.cpu().numpy()))).tolist()

        ds_weight = ds_weights[i]
        ds_weight = ds_weight[:, mask1, :, :]
        ds_weight = ds_weight[mask2, :, :, :]
        pruning_ds_weights.append(ds_weight)

        # 对bn层进行剪枝
        ds_bn_weight = ds_bn_weights[i]
        ds_bn_weight = ds_bn_weight[mask2]
        pruning_ds_bn_weights.append(ds_bn_weight)

        ds_bn_bias = ds_bn_biass[i]
        ds_bn_bias = ds_bn_bias[mask2]
        pruning_ds_bn_biass.append(ds_bn_bias)

        ds_bn_running_var = ds_bn_running_vars[i]
        ds_bn_running_var = ds_bn_running_var[mask2]
        pruning_ds_bn_running_vars.append(ds_bn_running_var)

        ds_bn_running_mean = ds_bn_running_means[i]
        ds_bn_running_mean = ds_bn_running_mean[mask2]
        pruning_ds_bn_running_means.append(ds_bn_running_mean)

    # ds卷积层剪枝后的参数保存到模型文件
    num1, num2, num3, num4, num5 = 0, 0, 0, 0, 0
    for k, v in ckpt.items():
        if 'downsample.0.weight' in k:
            ckpt[k] = pruning_ds_weights[num1]
            num1 = num1 + 1
        elif 'downsample.1.weight' in k:
            ckpt[k] = pruning_ds_bn_weights[num2]
            num2 = num2 + 1
        elif 'downsample.1.bias' in k:
            ckpt[k] = pruning_ds_bn_biass[num3]
            num3 = num3 + 1
        elif 'downsample.1.running_mean' in k:
            ckpt[k] = pruning_ds_bn_running_means[num4]
            num4 = num4 + 1
        elif 'downsample.1.running_var' in k:
            ckpt[k] = pruning_ds_bn_running_vars[num5]
            num5 = num5 + 1

    # print("remove_ratio: ", remove_ratio)
    # print("c_in_out: ", c_in_out)

    from model.backbone.resnet import pruned_Resnet
    new_model = pruned_Resnet(c_in_out=c_in_out)
    ckpt = {k.replace('encoder.', ''): v for k, v in ckpt.items()}
    # for k, v in new_model.state_dict().items():
    #     print("new_model: ", v.size())
    # for k, v in ckpt.items():
    #     print("ckpt: ", v.size())

    # new_model.load_state_dict(ckpt) # 保留未被剪枝的权重
    for m in new_model.modules():  # 或重新随机初始化
        if isinstance(m, nn.Conv2d):
            print("init Conv2d")
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            print("init BatchNorm2d")
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return new_model