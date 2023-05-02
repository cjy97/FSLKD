import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# from classifier import FewShotModel

__all__ = ["DN4"]

class DN4(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dn4_layer = DN4Layer(args.way, args.shot, args.query, n_k=3)

        # self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, instance_embs, support_idx, query_idx):

        way = self.args.way
        shot = self.args.shot
        query_num = self.args.query
        
        b, emb_dim, h, w = instance_embs.size()
        episode_size = b // (way * (shot + query_num))

        support = instance_embs[support_idx.contiguous().view(-1)].unsqueeze(0)
        query = instance_embs[query_idx.contiguous().view(-1)].unsqueeze(0)

        support = support.view(episode_size, shot, way, emb_dim, h, w)
        support = support.permute(0, 2, 1, 3, 4, 5)
        support = support.contiguous().view(episode_size, way * shot, emb_dim, h, w)

        logits = self.dn4_layer(query, support).view(episode_size * way * query_num, way) / self.args.temperature
        # logits = self.ff.mul_scalar(logits, 1.0/self.args.temperature )
        return logits


class DN4Layer(nn.Module):
    def __init__(self, way_num, shot_num, query_num, n_k):
        super(DN4Layer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.n_k = n_k

    def forward(self, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(t, self.way_num * self.query_num, c, h * w) \
            .permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=2).unsqueeze(2)
        # query_feat = query_feat.unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = support_feat.view(t, self.way_num, self.shot_num, c, h * w) \
            .permute(0, 1, 3, 2, 4).contiguous() \
            .view(t, self.way_num, c, self.shot_num * h * w)
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)
        # support_feat = support_feat.unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        # print("query_feat: ", query_feat.size())
        # print("query_feat: ", query_feat.dtype)
        # print("query_feat: ", query_feat)
        # print("support_feat: ", support_feat.size())

        relation = torch.matmul(query_feat, support_feat)
        # relation = My_quant_matmul(query_feat, support_feat)

        # print("relation: ", relation.size())        # [1, 75, 5, 25, 25]
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        # print("topk_value: ", topk_value.size())    # [1, 75, 5, 25, 3]
        score = torch.sum(topk_value, dim=[3, 4])
        # score = My_qunat_sum(topk_value, dim=[3, 4])

        # print("dn4 output score: ", score.size())   # [1, 75, 5]
        return score
