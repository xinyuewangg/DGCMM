import torch
import torch.nn as nn


class MMDLoss(nn.Module):

    def forward(self, sample_qz, sample_pz=None):
        if sample_pz is None:
            sample_pz = torch.randn_like(sample_qz)
        sigma2_p = 1. ** 2
        n = 100

        norms_pz = (sample_pz ** 2).sum(1, True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2. * dotprods_pz

        norms_qz = (sample_qz ** 2).sum(1, True)
        dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
        distances_qz = norms_qz + norms_qz.t() - 2. * dotprods_qz

        dotprods = torch.matmul(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2. * dotprods

        Cbase = 2. * 64 * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = res1 * (1. - torch.eye(len(res1), device=res1.device))
            res1 = res1.sum() / (n * n - n)
            res2 = C / (C + distances)
            res2 = res2.sum() * 2. / (n * n)
            stat += res1 - res2

        return stat
