from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FL import FTLoss
from .KD import DistillKL

class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""
    def __init__(self, args):
        super(SemCKDLoss, self).__init__()
        self.criterion_kl = DistillKL(args.temp)
        self.criterion_fl = FTLoss()
        self.crit = nn.MSELoss(reduction='none')
        self.loss_coefficient = args.loss_coefficient
        self.feature_loss_coefficient = args.feature_loss_coefficient
        
    def forward(self, s_value, f_target, weight):
        bsz, num_stu = weight.shape
        ind_loss_kd = torch.zeros(bsz, num_stu).cuda()
        ind_loss_fl = torch.zeros(bsz, num_stu).cuda()

        for i in range(num_stu):
            ind_loss_kd[:, i] = self.criterion_kl(s_value[i], f_target).reshape(bsz,-1).mean(-1)
            ind_loss_fl[:, i] = self.criterion_fl(s_value[i], f_target).reshape(bsz,-1).mean(-1)
            
        loss_kd = (weight * ind_loss_kd).sum()/(1.0*bsz)
        loss_fl = (weight * ind_loss_fl).sum()/(1.0*bsz)
        return self.loss_coefficient * loss_kd + self.feature_loss_coefficient * loss_fl