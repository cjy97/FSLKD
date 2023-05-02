import torch
from torch import nn

from Quant_module.My_quant_func import My_quant_sum

__all__ = ["RelationNet"]


class RelationNet(nn.Module):
    def __init__(self, args, feat_dim=640, feat_height=1, feat_width=1):
        super().__init__()

        if args.backbone == "Res12":
            self.feat_dim = 640
            self.feat_height = 1
            self.feat_width = 1
            pool = False
        elif args.backbone == "Vit":
            self.feat_dim = 768
            self.feat_height = 1
            self.feat_width = 1
            pool = True

        # self.feat_dim = feat_dim
        # self.feat_height = feat_height
        # self.feat_width = feat_width

        self.way = args.way
        self.shot = args.shot
        self.query_num = args.query

        self.relation_layer = RelationLayer(
            self.feat_dim, self.feat_height, self.feat_width, pool
        )
        # self.loss_func = nn.CrossEntropyLoss()

    def forward(self, instance_embs, support_idx, query_idx):

        b, emb_dim, h, w = instance_embs.size()
        episode_size = b // (self.way * (self.shot + self.query_num))

        support = instance_embs[support_idx.contiguous().view(-1)].unsqueeze(0)
        query = instance_embs[query_idx.contiguous().view(-1)].unsqueeze(0)

        # print("support: ", support.size())
        # print("query: ", query.size())
        support = support.view(episode_size, self.shot, self.way, emb_dim, h, w)
        support = support.permute(0, 2, 1, 3, 4, 5)
        support = support.contiguous().view(episode_size, self.way * self.shot, emb_dim, h, w)

        # query = query.view(episode_size, self.query_num, self.way, emb_dim, h, w)
        # query = query.permute(0, 2, 1, 3, 4, 5)
        # query = query.contiguous().view(episode_size, self.way * self.query_num, emb_dim, h, w)

        # print("support: ", support.size())
        # print("query: ", query.size())

        relation_pair = self._calc_pairs(query, support)

        logits = self.relation_layer(relation_pair).reshape(-1, self.way)

        return logits

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        relation_pair = self._calc_pairs(query_feat, support_feat)

        output = self.relation_layer(relation_pair).reshape(-1, self.way_num)

        # acc = accuracy(output, query_target.reshape(-1))
        return output  # , acc

    def _calc_pairs(self, query_feat, support_feat):
        """
        :param query_feat: (task_num, query_num * way_num, feat_dim, feat_width, feat_height)
        :param support_feat: (task_num, support_num * way_num, feat_dim, feat_width, feat_height)
        :return: query_num * way_num * way_num, feat_dim, feat_width, feat_height
        """
        t, _, c, h, w = query_feat.size()
        # t, w, wq, c, h, w -> t, wq, w, c, h, w
        query_feat = query_feat.unsqueeze(1).repeat(1, self.way, 1, 1, 1, 1)
        query_feat = torch.transpose(query_feat, 1, 2)

        # t, w, s, c, h, w -> t, 1, w, c, h, w -> t, wq, w, c, h, w
        support_feat = support_feat.reshape(t, self.way, self.shot, c, h, w)
        support_feat = (
            # torch.sum(support_feat, dim=(2,))
            #     .unsqueeze(1)
            #     .repeat(1, self.way * self.query_num, 1, 1, 1, 1)
            My_quant_sum(support_feat, dim=(2,), Int_Type=torch.int64)
                .unsqueeze(1)
                .repeat(1, self.way * self.query_num, 1, 1, 1, 1)
        )

        # t, wq, w, 2c, h, w -> twqw, 2c, h, w
        relation_pair = torch.cat((query_feat, support_feat), dim=3).reshape(-1, c * 2, h, w)

        return relation_pair


class RelationLayer(nn.Module):
    def __init__(self, feat_dim=64, feat_height=3, feat_width=3, pool=True):
        super(RelationLayer, self).__init__()

        if pool:
            self.layers = nn.Sequential(
                nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=0),
                nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=0),
                nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=0),
                nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(2),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=0),
                nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(2),
            )

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * feat_height * feat_width, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        # print(x.shape)
        # print("x: ", x.size())
        out = self.layers(x)
        # print("out: ", out.size())
        out = out.reshape(x.size(0), -1)
        # print("out: ", out.size())
        out = self.fc(out)
        return out