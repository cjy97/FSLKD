import torch
import torch.nn as nn
import numpy as np
from functools import partial

from model.backbone import Res12
from model.backbone import VisionTransformer

from model.classifier.protonet import ProtoNet
from model.classifier.dn4 import DN4
from model.classifier.relation_net import RelationNet

from Distill_module.distill_layer import DistillLayer
from Distill_module.Prune import prune_resnet

class FSLClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.backbone == "Res12":
            self.encoder = Res12()
        elif args.backbone == "Vit":
            self.encoder = VisionTransformer(img_size=84, patch_size=7, num_classes=64,
                                             embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
        else:
            raise ValueError('')

        if args.classifier == "ProtoNet":
            self.fsl_head = ProtoNet(args)
        elif args.classifier == "DN4":
            self.fsl_head = DN4(args)
        elif args.classifier == "RelationNet":
            self.fsl_head = RelationNet(args)
        else:
            raise ValueError('')

        if args.is_distill and args.kd_loss == "ACD":
        # 辅助分类器及GAP
            self.GAP = nn.AdaptiveAvgPool2d((1,1))
            if self.args.dataset == "MiniImageNet":
                self.fc = nn.Linear(640, 64)
            elif self.args.dataset == "TieredImageNet":
                self.fc = nn.Linear(640, 351)
            else:
                raise ValueError('')

        if args.is_distill:
            self.distill_layer = DistillLayer(args).requires_grad_(False)

        if args.is_prune:  # 如果指定了“剪枝”模式，则重置学生模型为Slim-Resnet12
            self.encoder = prune_resnet(self.encoder, args.remove_ratio)
    
    def split_instances(self):
        args = self.args
        return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way),
                 torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))


    def forward(self, x):

        # if forward_mode:
        if x.size()[0] == 6:    # for Forward test
            support_idx = torch.arange(5).reshape(1, 1, 5)
            query_idx = torch.tensor([[[5]]])
            query_idx = query_idx.repeat(1, 15, 5)
        else:
            support_idx, query_idx = self.split_instances()
        # print("support_idx: ", support_idx)
        # print("query_idx: ", query_idx)

        x = x.squeeze(0)

        instance_embs = self.encoder(x)
        logits = self.fsl_head(instance_embs, support_idx, query_idx)

        if self.training:

            if self.args.is_distill is False:  # (仅当is_distill为false)
                return logits

            if self.args.kd_loss == "KD":   # 直接基于小样本分类概率计算蒸馏损失
                teacher_instance_embs = self.distill_layer(x)

                teacher_logits = self.fsl_head(teacher_instance_embs, support_idx, query_idx)
                return logits, teacher_logits

            elif self.args.kd_loss == "ACD":
                teacher_logits = self.distill_layer(x)

                features = self.GAP(instance_embs)
                student_logits = self.fc(features.view(features.size(0), -1))

                return logits, student_logits, teacher_logits

            elif self.args.kd_loss == "LFD" or self.args.kd_loss == "IRD":
                student_encoding = instance_embs
                teacher_encoding = self.distill_layer(x)

                return logits, student_encoding, teacher_encoding

        else:
            return logits


# class FewShotModel(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         # if args.backbone_class == 'ConvNet':
#         #     from model.networks.convnet import ConvNet
#         #     self.encoder = ConvNet()
#         # elif args.backbone_class == 'Res12':
#         #     hdim = 640
#         #     from model.networks.res12 import ResNet
#         #     self.encoder = ResNet()
#         # elif args.backbone_class == 'Res18':
#         #     hdim = 512
#         #     from model.networks.res18 import ResNet
#         #     self.encoder = ResNet()
#         # elif args.backbone_class == 'WRN':
#         #     hdim = 640
#         #     from model.networks.WRN28 import Wide_ResNet
#         #     self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
#         # elif args.backbone_class == "ViT":
#         #     hdim = 768
#         #     from model.networks.vit import ViT
#         #     self.encoder = ViT()
#         if args.backbone == "Res12":
#             from backbone import Res12
#             self.encoder = Res12()
#         else:
#             raise ValueError('')

#     def split_instances(self, data):
#         args = self.args
#         return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way),
#                  torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))


#     def forward(self, x, get_feature=False):
#         if get_feature:
#             # get feature with the provided embeddings
#             return self.encoder(x)
#         else:
#             # feature extraction
#             x = x.squeeze(0)
#             # instance_embs = self.encoder(x)
#             # num_inst = instance_embs.shape[0]
#             # split support query set for few-shot data
#             support_idx, query_idx = self.split_instances(x)
#             if self.training:
#                 # logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
#                 result = self._forward(x, support_idx, query_idx)
#                 return result
#             else:
#                 # logits = self._forward(instance_embs, support_idx, query_idx)
#                 logits = self._forward(x, support_idx, query_idx)
#                 return logits

#     def _forward(self, x, support_idx, query_idx):
#         raise NotImplementedError('Suppose to be implemented by subclass')