import torch
import torch.nn.functional as F
import torch.nn as nn


mse_loss = nn.MSELoss(size_average=False, reduce=True)
GAP = nn.AvgPool2d(5, stride=1)


class Local_KD(nn.Module):
    def __init__(self, way_num, shot_query):
        super(Local_KD, self).__init__()

        self.way_num = way_num
        self.shot_query = shot_query
        # self.n_k = n_k
        # self.device = device
        # self.normLayer = nn.BatchNorm1d(self.way_num * 2, affine=True)
        # self.fcLayer = nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)

    # def _cal_cov_matrix_batch(self, feat):
    #     e, _, n_local, c = feat.size()
    #     feature_mean = torch.mean(feat, 2, True)
    #     feat = feat - feature_mean
    #     cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
    #     cov_matrix = torch.div(cov_matrix, n_local-1)
    #     cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

    #     return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):
        e, b, c, h, w = feat.size()
        feat = feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()  # e, b, h*w, c
        feat = feat.view(e, 1, b * h * w, c)  # e, 1, b*h*w, c
        feat_mean = torch.mean(feat, 2, True)  # e, 1, 1, c
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)  # e, 1, c, c
        cov_matrix = torch.div(cov_matrix, b * h * w - 1)  # b*h*w !!
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).cuda()  # to(self.device)

        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        print("mean1: ", mean1.size())  # [e, 1, 1, 640]
        print("cov1: ", cov1.size())  # [e, 1, 640, 640]
        print("mean2: ", mean2.size())  # [e, 1, 1, 640]
        print("cov2: ", cov2.size())  # [e, 1, 640, 640]

        cov2_inverse = torch.inverse(cov2)
        mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))

        matrix_prod = torch.matmul(
            cov1.unsqueeze(2), cov2_inverse.unsqueeze(1)
        )
        # print("matrix_prod: ", matrix_prod.size())

        trace_dist = torch.diagonal(matrix_prod, offset=0, dim1=-2, dim2=-1)
        trace_dist = torch.sum(trace_dist, dim=-1)
        # print("trace_dist: ", trace_dist.size())

        maha_prod = torch.matmul(
            mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1)
        )
        maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(4))
        maha_prod = maha_prod.squeeze(4)
        maha_prod = maha_prod.squeeze(3)

        matrix_det = torch.slogdet(cov2).logabsdet.unsqueeze(1) - torch.slogdet(
            cov1
        ).logabsdet.unsqueeze(2)

        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)

        return kl_dist / 2.0


def KD_loss_func(args, results):

    # 无蒸馏
    if args.is_distill is False:
        logits = results
        kd_loss = 0.0
        return logits, kd_loss


    if args.kd_loss == "KD":
        logits, teacher_logits = results

        if teacher_logits is not None:
            T = 4.0
            p_s = F.log_softmax(logits / T, dim=1)
            p_t = F.softmax(teacher_logits)
            # print("p_s: ", p_s)
            # print("p_t: ", p_t)
            kd_loss = F.kl_div(
                p_s,
                p_t,
                reduction='sum'
                # size_average=False
            )  # * (T**2)
        else:
            kd_loss = 0.0

    elif args.kd_loss == "ACD":
        logits, student_logits, teacher_logits = results

        # print("student_logits: ", student_logits.size())
        # print("teacher_logits: ", teacher_logits.size())
        if teacher_logits is not None:
            T = 4.0
            p_s = F.log_softmax(student_logits / T, dim=1)
            p_t = F.softmax(teacher_logits)
            # print("p_s: ", p_s)
            # print("p_t: ", p_t)
            kd_loss = F.kl_div(
                p_s,
                p_t,
                reduction='sum'
                # size_average=False
            )  # * (T**2)
        else:
            kd_loss = 0.0

    elif args.kd_loss == "global_KD":   # 全局特征蒸馏
        logits, student_encoding, teacher_encoding = results
        # print("student_encoding: ", student_encoding.size())
        # print("teacher_encoding: ", teacher_encoding.size())
        if teacher_encoding is not None:
            student_feat = student_encoding
            teacher_feat = teacher_encoding

            student_feat = GAP(student_encoding).view(student_feat.size(0), -1).unsqueeze(0)
            teacher_feat = GAP(teacher_encoding).view(teacher_feat.size(0), -1).unsqueeze(0)

            print("student_feat: ", student_feat.size())
            print("teacher_feat: ", teacher_feat.size())

            T = 4.0
            p_s = F.log_softmax(student_feat / T, dim=1)
            p_t = F.softmax(teacher_feat)
            kd_loss = F.kl_div(
                p_s,
                p_t,
                # reduction='sum'
                size_average=True
            )  # * (T**2)

            # dim不一致如何解决？

        else:
            kd_loss = 0.0

    elif args.kd_loss == "relation_KD":     # 样本间关系蒸馏
        logits, student_encoding, teacher_encoding = results

        if teacher_encoding is not None:
            student_feat = student_encoding
            teacher_feat = teacher_encoding

            student_feat = GAP(student_encoding).view(student_feat.size(0), -1).unsqueeze(0)
            teacher_feat = GAP(teacher_encoding).view(teacher_feat.size(0), -1).unsqueeze(0)

            teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1),
                                            torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))
            student_relation = torch.matmul(F.normalize(student_feat, p=2, dim=-1),
                                            torch.transpose(F.normalize(student_feat, p=2, dim=-1), -1, -2))

            kd_loss = mse_loss(teacher_relation, student_relation)

        else:
            kd_loss = 0.0

    # elif args.kd_loss == "local_KD":
    #     logits, student_encoding, teacher_encoding = results
    #
    #     student_feat = student_encoding.unsqueeze(0)
    #     teacher_feat = teacher_encoding.unsqueeze(0)
    #
    #     local_kd = Local_KD(args.way, args.shot + args.query)
    #     kd_loss = local_kd(student_feat, teacher_feat)

    elif args.kd_loss == "LFD":
        logits, student_encoding, teacher_encoding = results

        student_feat = student_encoding
        teacher_feat = teacher_encoding
        # print("student_feat: ", student_feat.size())    # [80, 640, 5, 5]
        # print("teacher_feat: ", teacher_feat.size())
        # 这里本来应该将形如[bs, emb_dim, h, w]的feat数据变形成[bs*h*w, emb_dim]的形式，但经测试不转换其实对结果没有影响

        T = 4.0
        p_s = F.log_softmax(student_feat / T, dim=1)
        p_t = F.softmax(teacher_feat / T, dim=1)
        kd_loss = F.kl_div(
            p_s,
            p_t,
            size_average=False
        ) * (T ** 2)

    elif args.kd_loss == "IRD":
        logits, student_encoding, teacher_encoding = results

        b, emb_dim, h, w = student_encoding.size()

        student_feat = student_encoding
        teacher_feat = teacher_encoding

        student_feat = student_feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, emb_dim)
        teacher_feat = teacher_feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, emb_dim)

        student_relation = torch.matmul(F.normalize(student_feat, p=2, dim=-1),
                                        torch.transpose(F.normalize(student_feat, p=2, dim=-1), -1, -2))

        teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1),
                                        torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))
        # print("teacher_relation matrix: ", teacher_relation.size()) # [80, 25, 25]
        # print("teacher_relation matrix: ", teacher_relation)

        # criterion = nn.L1Loss(size_average=False, reduce=True)
        criterion = nn.MSELoss(size_average=False, reduce=True)
        kd_loss = criterion(teacher_relation, student_relation)

    elif args.kd_loss == "ALL":
        logits, student_logits, teacher_logits, student_encoding, teacher_encoding = results
        b, emb_dim, h, w = student_encoding.size()

        T = 4.0
        p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits)
        loss_KD = F.kl_div(
            p_s,
            p_t,
            reduction='sum'
            # size_average=False
        )  # * (T**2)

        student_feat = student_encoding
        teacher_feat = teacher_encoding
        T = 4.0
        p_s = F.log_softmax(student_feat / T, dim=1)
        p_t = F.softmax(teacher_feat / T, dim=1)
        loss_local_kd_pos = F.kl_div(
            p_s,
            p_t,
            size_average=False
        ) * (T ** 2)

        student_feat = student_encoding
        teacher_feat = teacher_encoding
        student_feat = student_feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, emb_dim)
        teacher_feat = teacher_feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, emb_dim)
        student_relation = torch.matmul(F.normalize(student_feat, p=2, dim=-1),
                                        torch.transpose(F.normalize(student_feat, p=2, dim=-1), -1, -2))
        teacher_relation = torch.matmul(F.normalize(teacher_feat, p=2, dim=-1),
                                        torch.transpose(F.normalize(teacher_feat, p=2, dim=-1), -1, -2))
        criterion = nn.MSELoss(size_average=False, reduce=True)
        loss_local_kd_rel = criterion(teacher_relation, student_relation)

        kd_loss = loss_KD * 1.0 + loss_local_kd_pos * 1.0 + loss_local_kd_rel * 1.0

    else:
        logits = results[0]
        kd_loss = 0.0

    return logits, kd_loss