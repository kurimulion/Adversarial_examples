import torch
import torch.nn as nn
import math

# the module is taken from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        y_ = y.type(torch.cuda.LongTensor)
        gamma, beta = self.embed(y_).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
    
class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        y_ = y.type(torch.cuda.LongTensor)
        gamma, beta = self.embed(y_).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

# This module implement the self-attention block in https://arxiv.org/abs/1805.08318
class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.channel_in = in_dim

        self.f = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.g = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.softmax  = nn.Softmax(dim=-1)
        self.h = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 2, 1))
        self.pool = nn.MaxPool2d(kernel_size = 2, ceil_mode=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.v = nn.utils.spectral_norm(nn.Conv2d(in_dim // 2, in_dim, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        location_num = H * W
        downsampled_num = math.ceil(H / 2) * math.ceil(W / 2)

        f = self.f(x).view(N, -1, H * W).permute(0, 2, 1)   # N, H*W, C
        g = self.g(x)
        g = self.pool(g).view(N, -1, downsampled_num)        # N, C, (H // 2 + 1) * (W // 2 + 1)

        attn = torch.bmm(f, g)
        attn = self.softmax(attn)                           # N, H*W, H*W//4

        h = self.h(x)
        h = self.pool(h).view(N, -1, downsampled_num).permute(0, 2, 1)   # N, (H // 2 + 1) * (W // 2 + 1), C//2

        attn = torch.bmm(attn, h).view(N, -1, H, W)         # N, C//2, H, W
        out = x + self.v(attn) * self.gamma
        return out
