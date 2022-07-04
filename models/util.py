from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feat_dim, s_n, t_n, soft, factor=8):
        super(SelfAttention, self).__init__()
        self.s_len = len(s_n)
        self.feat_dim = feat_dim
        self.soft = soft
        setattr(self, 'query_weight_teacher', Embed(t_n, feat_dim))
        self.linear = nn.Linear(self.feat_dim ** 2, self.feat_dim ** 2 // factor)
        for i in range(self.s_len):
            setattr(self, 'key_weight' + str(i), Embed(s_n[i], feat_dim))

        for i in range(self.s_len):
            setattr(self, 'regressor' + str(i), Proj(s_n[i], t_n))

    def forward(self, f_s, f_t):
        # feature space alignment
        proj_value_stu = []
        for i in range(self.s_len):
            # proj_value_stu.append
            s_H, t_H = f_s[i].shape[2], f_t.shape[2]
            if s_H > t_H:
                f_s[i] = F.adaptive_avg_pool2d(f_s[i], (t_H, t_H))
                source = f_s[i]
                target = f_t
            elif s_H < t_H or s_H == t_H:
                f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
                target = f_t
                source = f_s[i]
            proj_value_stu.append(getattr(self, 'regressor' + str(i))(source))
        value_tea = target
        f_t = self.query_weight_teacher(f_t)
        bsz, ch = f_t.shape[0], f_t.shape[1]
        for i in range(self.s_len):
            f_s[i] = getattr(self, 'key_weight' + str(i))(f_s[i])
            f_s[i] = f_s[i].view(bsz, -1)
        f_t = f_t.view(bsz, -1)
        # emd_t = torch.bmm(f_t, f_t.permute(0,2,1))
        # emd_t = torch.nn.functional.normalize(emd_t, dim = 2)
        # emd_t  = emd_t.view(bsz, -1)
        # emd_t = self.linear(emd_t)
        # emd_s = list(range(self.s_len))
        '''
        for i in range(self.s_len):
            emd_s[i] = torch.bmm(f_s[i], f_s[i].permute(0,2,1))
            emd_s[i] = torch.nn.functional.normalize(emd_s[i],dim=2)
            emd_s[i] = emd_s[i].view(bsz, -1)
            emd_s[i] = self.linear(emd_s[i])
        '''
        # query of target layers
        proj_query = f_t
        proj_query = proj_query[:, None, :]

        # key of source layers   
        proj_key = f_s[0]
        proj_key = proj_key[:, :, None]
        for i in range(1, len(f_s)):
            temp_proj_key = f_s[i]
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

        # attention weight
        energy = torch.bmm(proj_query, proj_key) / self.soft  # batch_size X No.Tea feature X No.Stu feature
        attention = F.softmax(energy, dim=-1)
        attention = attention.squeeze(dim=1)

        return proj_value_stu, value_tea, attention


class AAEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)  # Normalize(2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
