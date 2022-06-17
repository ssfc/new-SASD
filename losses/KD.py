from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    def __init__(self, temp_factor):
        super(DistillKL, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="none")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss