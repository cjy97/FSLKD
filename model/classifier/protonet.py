import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


__all__ = ["ProtoNet"]

class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        # self.GAP = nn.AvgPool2d(5, stride=1)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, instance_embs, support_idx, query_idx):

        # print("instance_embs： ", instance_embs.size())
        features = self.GAP(instance_embs)
        # print("features: ", features.size())
        features = features.view(features.size(0), -1)

        emb_dim = features.size(-1)

        # organize support/query data
        support = features[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = features[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if True:  # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature

        else:  # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)


        return logits
