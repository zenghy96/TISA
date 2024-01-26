import torch
from torch import nn
import torch.nn.functional as F
from .evidence_loss import calculate_evidential_loss_constraints


class Hourglass(nn.Module):
    """
    basic hourglass
    """
    def __init__(self, n, f):
        super(Hourglass, self).__init__()
        self.up = Residual(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, f)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, f)
        else:
            self.low2 = Residual(f, f)
        self.low3 = Residual(f, f)
        self.low = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up = self.up(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        low = self.low(low3)
        #print('hourgalss:',x.shape)
        return up + low


class Residual(nn.Module):
    """
    residual model
    """
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        # batch normalize,ReLu-->convolution-->batch normalize,ReLu
        self.ins = ins
        self.outs = outs
        self.LowBranch = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins, outs//2, kernel_size=1),  # 1X1 kernel for reducing parameter
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2, outs//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs//2, outs, kernel_size=1))
            # nn.ReLU(inplace=True))
        # skip layer
        if ins != outs:
            self.UpBranch = nn.Conv2d(ins, outs, 1)

    def forward(self, x):
        if self.ins != self.outs:
            res = self.UpBranch(x)
        else:
            res = x
        x = self.LowBranch(x)
        x = x + res
        #print('Res:',x.shape)
        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        #print('Conv:',x.shape)
        return x


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class StackedHourglass(nn.Module):
    def __init__(self, stacks, hg_order, inp_dim, oup_dim):
        super(StackedHourglass, self).__init__()

        self.stacks = stacks
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=True, relu=True),
            nn.MaxPool2d(2, 2),
        )
        self.res = nn.Sequential(
            Residual(64, 128),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(hg_order, inp_dim),
            ) for i in range(stacks)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(stacks)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(stacks)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(stacks - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(stacks - 1)])
        self.stacks = stacks

        self.transform_v = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(stacks)])
        self.transform_alpha = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(stacks)])
        self.transform_beta = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(stacks)])

        self._ev_dec_v_max = 20
        self._ev_dec_alpha_max = 20
        self._ev_dec_beta_min = 0.2

    def forward(self, x, heatmaps=None):
        x = self.pre(x)
        x = self.res(x)
        combined_preds = []
        combined_v = []
        combined_alpha = []
        combined_beta = []

        for i in range(self.stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)

            # ######################### evidence ####################
            pred = self.outs[i](feature)
            logv = self.transform_v[i](feature)
            logalpha = self.transform_alpha[i](feature)
            logbeta = self.transform_beta[i](feature)

            v = F.softplus(logv)
            alpha = F.softplus(logalpha) + 1
            beta = F.softplus(logbeta)

            alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha.shape).to(alpha.device)
            alpha = torch.min(alpha, alpha_thr)
            v_thr = self._ev_dec_v_max * torch.ones(v.shape).to(v.device)
            v = torch.min(v, v_thr)
            beta_min = self._ev_dec_beta_min * torch.ones(beta.shape).to(beta.device)
            beta = beta + beta_min
            # #####################################################

            if i < self.stacks - 1:
                x = x + self.merge_preds[i](pred) + self.merge_features[i](feature)

            combined_preds.append(pred)
            combined_v.append(v)
            combined_alpha.append(alpha)
            combined_beta.append(beta)

        if heatmaps is not None:
            loss = []
            for ii in range(self.stacks):
                evidential_loss, logging_dict = calculate_evidential_loss_constraints(
                    0,
                    y=heatmaps,
                    mu=combined_preds[ii],
                    v=combined_v[ii],
                    alpha=combined_alpha[ii],
                    beta=combined_beta[ii]
                )
                loss.append(evidential_loss)
            loss = torch.stack(loss, 0).mean()
            return combined_preds, combined_v, combined_alpha, combined_beta, loss
        else:
            return combined_preds, combined_v, combined_alpha, combined_beta
