import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class _CrossNeuronBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width,
                    nblocks_channel=4,
                    spatial_height=24, spatial_width=24,
                    reduction=8, size_is_consistant=True):
        # nblock_channel: number of block along channel axis
        # spatial_size: spatial_size
        super(_CrossNeuronBlock, self).__init__()

        # set channel splits
        if in_channels <= 512:
            self.nblocks_channel = 1
        else:
            self.nblocks_channel = in_channels // 512
        block_size = in_channels // self.nblocks_channel
        block = torch.Tensor(block_size, block_size).fill_(1)
        self.mask = torch.Tensor(in_channels, in_channels).fill_(0)
        for i in range(self.nblocks_channel):
            self.mask[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size].copy_(block)

        # set spatial splits
        if in_height * in_width < 32 * 32 and size_is_consistant:
            self.spatial_area = in_height * in_width
            self.spatial_height = in_height
            self.spatial_width = in_width
        else:
            self.spatial_area = spatial_height * spatial_width
            self.spatial_height = spatial_height
            self.spatial_width = spatial_width

        self.fc_in = nn.Sequential(
            nn.Conv1d(self.spatial_area, self.spatial_area // reduction, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(self.spatial_area // reduction, self.spatial_area, 1, 1, 0, bias=True),
        )

        self.fc_out = nn.Sequential(
            nn.Conv1d(self.spatial_area, self.spatial_area // reduction, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(self.spatial_area // reduction, self.spatial_area, 1, 1, 0, bias=True),
        )

        self.bn = nn.BatchNorm1d(self.spatial_area)

    def forward(self, x):
        '''
        :param x: (bt, c, h, w)
        :return:
        '''
        bt, c, h, w = x.shape
        residual = x
        x_stretch = x.view(bt, c, h * w)
        spblock_h = int(np.ceil(h / self.spatial_height))
        spblock_w = int(np.ceil(w / self.spatial_width))
        stride_h = int((h - self.spatial_height) / (spblock_h - 1)) if spblock_h > 1 else 0
        stride_w = int((w - self.spatial_width) / (spblock_w - 1)) if spblock_w > 1 else 0

        # import pdb; pdb.set_trace()

        if spblock_h == 1 and spblock_w == 1:
            x_stacked = x_stretch # (b) x c x (h * w)
            x_stacked = x_stacked.view(bt * self.nblocks_channel, c // self.nblocks_channel, -1)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b) x (h * w) x c
            x_v = self.fc_in(x_v) # (b) x (h * w) x c
            x_m = x_v.mean(1).view(-1, 1, c // self.nblocks_channel).detach() # (b * h * w) x 1 x c
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * h * w) x c x c
            # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * h * w) x c x c
            out = self.bn(self.fc_out(torch.bmm(x_v, attn))) # (b) x (h * w) x c
            out = out.permute(0, 2, 1).contiguous().view(bt, c, h, w)
            return F.relu(residual + out)
        else:
            # first splt input tensor into chunks
            ind_chunks = []
            x_chunks = []
            for i in range(spblock_h):
                for j in range(spblock_w):
                    tl_y, tl_x = max(0, i * stride_h), max(0, j * stride_w)
                    br_y, br_x = min(h, tl_y + self.spatial_height), min(w, tl_x + self.spatial_width)
                    ind_y = torch.arange(tl_y, br_y).view(-1, 1)
                    ind_x = torch.arange(tl_x, br_x).view(1, -1)
                    ind = (ind_y * w + ind_x).view(1, 1, -1).repeat(bt, c, 1).type_as(x_stretch).long()
                    ind_chunks.append(ind)
                    chunk_ij = torch.gather(x_stretch, 2, ind).contiguous()
                    x_chunks.append(chunk_ij)

            x_stacked = torch.cat(x_chunks, 0) # (b * nb_h * n_w) x c x (b_h * b_w)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b * nb_h * n_w) x (b_h * b_w) x c
            x_v = self.fc_in(x_v) # (b * nb_h * n_w) x (b_h * b_w) x c
            x_m = x_v.mean(1).view(-1, 1, c) # (b * nb_h * n_w) x 1 x c
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * nb_h * n_w) x c x c
            score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * nb_h * n_w) x c x c
            out = self.bn(self.fc_out(torch.bmm(x_v, attn))) # (b * nb_h * n_w) x (b_h * b_w) x c

            # put back to original shape
            out = out.permute(0, 2, 1).contiguous() # (b * nb_h * n_w)  x c x (b_h * b_w)
            # x_stretch_out = x_stretch.clone().zero_()
            for i in range(spblock_h):
                for j in range(spblock_w):
                    idx = i * spblock_w + j
                    ind = ind_chunks[idx]
                    chunk_ij = out[idx * bt:(idx+1) * bt]
                    x_stretch = x_stretch.scatter_add(2, ind, chunk_ij / spblock_h / spblock_h)
            return F.relu(x_stretch.view(residual.shape))


class HSBlock(nn.Module):
    def __init__(self, in_planes, s, w):
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        if in_planes % s == 0:
            in_ch, in_ch_last = in_planes // s, in_planes // s
        else:
            in_ch, in_ch_last = (in_planes // s) + 1, in_planes - (in_planes // s + 1) * (s-1)
            # print(in_ch, in_ch_last)
        for i in range(self.s):
            if i == 0:
                self.module_list.append(nn.Sequential())
            elif i == 1:
                self.module_list.append(self.conv_bn_relu(in_ch=in_ch, out_ch=w))
            elif i == s - 1:
                self.module_list.append(self.conv_bn_relu(in_ch=in_ch_last + w // 2, out_ch=w))
            else:
                self.module_list.append(self.conv_bn_relu(in_ch=in_ch + w // 2, out_ch=w))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape, 'x')
        x = list(x.chunk(chunks=self.s, dim=1))
        # ttt = x[1]
        # print(ttt)
        # print(x, 'xxxx')
        for i in range(1, len(self.module_list)):
            # print(i, 'iii')
            # print(self.module_list[i], '11111')
            y = self.module_list[i](x[i])
            # print(y.shape)
            if i == len(self.module_list) - 1:
                # print(i, 'iiiiiiii')
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
            # print(x[0].shape)
        return x[0]


class AdaptiveReweight(nn.Module):
    def __init__(self, channel, reduction=4, momentum=0.1, index=0):
        self.channel = channel
        super(AdaptiveReweight, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LayerNorm([channel // reduction]),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.register_buffer('running_scale', torch.zeros(1))
        self.momentum = momentum
        self.ind = index

    def forward(self, x):
        b, c, _, _ = x.size()
        _x = x.view(b, c, -1)
        x_var = _x.var(dim=-1)

        y = self.fc(x_var).view(b, c)

        if self.training:
            scale = x_var.view(-1).mean(dim=-1).sqrt()
            self.running_scale.mul_(1. - self.momentum).add_(scale.data * self.momentum)
        else:
            scale = self.running_scale
        inv = (y / scale).view(b, c, 1, 1)
        return inv.expand_as(x) * x


class CE(nn.Module):
    def __init__(self, num_features, pooling=False, num_groups=1, num_channels=64, T=3, dim=4, eps=1e-5, momentum=0,
                 *args, **kwargs):
        super(CE, self).__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.dim = dim

        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
                                                                                                          num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        self.AR = AdaptiveReweight(num_features)
        self.pool = None
        if pooling:
            self.pool = nn.MaxPool2d(2, stride=2)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))

        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.x_weight = nn.Parameter(torch.zeros(1))
        print(self.num_channels)

    def forward(self, X):
        N, C, H, W = X.size()
        xin = self.AR(X)
        x_pool = self.pool(X) if self.pool is not None else X

        x_pool = x_pool.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        x = X.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        _, d, m = x.size()

        if self.training:
            mean = x_pool.mean(-1, keepdim=True)

            xc = x_pool - mean

            P = [None] * (self.T + 1)
            P[0] = torch.eye(d, device=X.device).expand(self.num_groups, d, d)
            Sigma = torch.baddbmm(alpha=self.eps, input=P[0], beta=1. / m, batch1=xc, batch2=xc.transpose(1, 2))

            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            Sigma_N = Sigma * rTr
            for k in range(self.T):
                mat_power3 = torch.matmul(torch.matmul(P[k], P[k]), P[k])
                P[k + 1] = torch.baddbmm(alpha=1.5, input=P[k], beta=-0.5, batch1=mat_power3, batch2=Sigma_N)

            wm = P[self.T]

            self.running_mean.mul_(1. - self.momentum).add_(mean.data * self.momentum)
            self.running_wm.mul_((1. - self.momentum)).add_(self.momentum * wm.data)
        else:
            xc = x - self.running_mean
            wm = self.running_wm

        xn = wm.matmul(x)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()

        x_weight = torch.sigmoid(self.x_weight)
        return x_weight * Xn + (1 - x_weight) * xin


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(1, 0), dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, temperature):
        super(AttentionGate, self).__init__()
        kernel_size = (5, 1)
        self.temperature = temperature
        self.compress = ZPool()
        # self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(2, 0), relu=False)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        # print(x.shape, 'ty1')
        x_compress = self.compress(x)
        # print(x_compress.shape, 'Z_pooling')
        x_out = self.conv(x_compress)
        # print(x_out.shape, 'Conv+BN+RelU')
        # scale = torch.softmax(x_out/self.temperature, 1)
        scale = torch.sigmoid(x_out)
        # print((x*scale).shape, 'ty4')
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, temperature=34):
        super(TripletAttention, self).__init__()

        self.cw = AttentionGate(temperature)
        self.hc = AttentionGate(temperature)
        self.no_spatial = no_spatial

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # initialization
        # self.w1 = torch.nn.init.normal_(self.w1)
        # self.w2 = torch.nn.init.normal_(self.w2)
        # self.w3 = torch.nn.init.normal_(self.w3)
        self.w1.data.fill_(1/3)
        self.w2.data.fill_(1/3)
        self.w3.data.fill_(1/3)

        if not no_spatial:
            self.hw = AttentionGate(temperature)

    def update_temperature(self):
        self.cw.updata_temperature()
        self.hc.updata_temperature()
        self.hw.updata_temperature()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        # print(x_out1.shape, 'ty44')
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            # print(x_out.shape, 'ty55')
            # x_out = x_out11
            # x_out = 1/3 * (x_out + x_out11 + x_out21)
            # x_out = 4 * x_out + 5 * x_out11 + 6 * x_out21
            x_out = self.w1 * x_out + self.w2 * x_out11 + self.w3 * x_out21
            # print(self.w1, self.w2, self.w3, 'w1,w2,w3')
            # print(x_out.shape, 'ty22')
        else:
            x_out = self.w1 * x_out11 + self.w2 * x_out21
        # return x_out, self.w1, self.w2, self.w3
        return x_out



