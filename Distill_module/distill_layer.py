import copy
import torch
import torch.nn as nn

from model.backbone.resnet import Res12

class DistillLayer(nn.Module):
    def __init__(self, args):
        super(DistillLayer, self).__init__()
        self.encoder = self._load_state_dict(args, type="encoder.")

        if args.kd_loss == "ACD" or args.kd_loss == "ALL":  # 只有基于辅助分类logits的蒸馏方法或全部损失方法并用时，教师模型需要加载线性分类头
            self.fc = self._load_state_dict(args, type="fc.")
            self.GAP = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.fc = None

    def _load_state_dict(self, args, type):
        new_model = None

        if args.is_distill and args.teacher_init_weights is not None:

            if type == "encoder.":
                if args.teacher_backbone_class == 'Res12':
                    new_model = Res12()
            elif type == "fc.":
                if args.dataset == "MiniImageNet":
                    new_model = nn.Linear(640, 64)
                elif args.dataset == "TieredImageNet":
                    new_model = nn.Linear(640, 351)

            model_dict = new_model.state_dict()

            pretrained_dict = torch.load(args.teacher_init_weights, map_location=torch.device('cpu'))['params']
            pretrained_dict = {k.replace(type, ""): v for k, v in pretrained_dict.items() if
                               k.replace(type, "") in model_dict}

            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict)  # 只将权重加载给教师模型

            for k, _ in pretrained_dict.items():
                print("pretrained key: ", k)
            for key, _ in pretrained_dict.items():
                print("key: ", key)

        return new_model

    @torch.no_grad()
    def forward(self, x):
        local_features = None

        if self.encoder is not None:
            local_features = self.encoder(x)

        if self.fc is not None:
            features = self.GAP(local_features)
            logits = self.fc(features.view(features.size(0), -1))
            return logits
        else:
            return local_features
