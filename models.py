import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, out_channels=1, conv_dim=64, MNIST=True):
        """
        This is a conditional discriminator with projection. The input is input image or generated image.
        With MNIST data the dimensions of output of last conv layer is N, conv_dim * 4, 1, 1
        Only implement the case for number of layer = 1, because the dimensions of the MNIST are much smaller.
        The code is adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
        """
        super(Discriminator, self).__init__()
        p = 1
        if MNIST:
            p += 2 
        sequence = [nn.utils.spectral_norm(nn.Conv2d(in_channels, conv_dim, 4, padding=p, stride=2)), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 5):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [nn.utils.spectral_norm(nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, 4, padding=1, stride=2)),
                         nn.BatchNorm2d(conv_dim * nf_mult),
                         nn.LeakyReLU(0.2)]
        self.model = nn.Sequential(*sequence)
        self.f1 = nn.utils.spectral_norm(nn.Linear(conv_dim * 4, 1))
        self.snembed = nn.utils.spectral_norm(nn.Embedding(num_classes, conv_dim * 4))

    def forward(self, x, y):
        x_ = self.model(x)
        x_ = x_.squeeze()
        x_0 = x_.clone()
        output = self.f1(x_)

        y_ = y.type(torch.cuda.LongTensor)
        labels = self.snembed(y_)
        output += torch.sum(x_0 * labels, 1, keepdim=True)
        #output = torch.sigmoid(output)

        return output

def Generator(in_channels=1, conv_dim=64, G_type='unet', norm='batch', MNIST=True):
    if norm == 'batch':
        norm_layer = utils.ConditionalBatchNorm2d
    else:
        norm_layer = utils.ConditionalInstanceNorm2d

    if G_type == 'unet':
        net = UnetGenerator(in_channels, norm_layer, conv_dim=conv_dim, MNIST=MNIST)
    else:
        net = ResnetGenerator(in_channels, norm_layer, conv_dim=conv_dim, MNIST=MNIST)
    return net

class ResnetGenerator(nn.Module):
    '''
    The code is adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    def __init__(self, in_channels=1, norm_layer=utils.ConditionalBatchNorm2d, 
                 use_dropout=False, num_classes=10, conv_dim=64, MNIST=True):
        super(ResnetGenerator, self).__init__()
        # affine transformation in normalization layer
        use_bias = False
        p = 1
        if MNIST:
            p += 2
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, conv_dim, 7, bias=use_bias))
        self.norm1 = norm_layer(conv_dim, num_classes)

        self.down1 = nn.utils.spectral_norm(nn.Conv2d(conv_dim, conv_dim * 2, 3, padding=p, stride=2, bias=use_bias))
        self.downnorm1 = norm_layer(conv_dim * 2, num_classes)
        self.down2 = nn.utils.spectral_norm(nn.Conv2d(conv_dim * 2, conv_dim * 4, 3, padding=1, stride=2, bias=use_bias))
        self.downnorm2 = norm_layer(conv_dim * 4, num_classes)
        self.down3 = nn.utils.spectral_norm(nn.Conv2d(conv_dim * 4, conv_dim * 8, 3, padding=1, stride=2, bias=use_bias))
        self.donwnrom3 = norm_layer(conv_dim * 8, num_classes)

        self.res1 = Resnet(conv_dim * 8, norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res2 = Resnet(conv_dim * 8, norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res3 = Resnet(conv_dim * 8, norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.res4 = Resnet(conv_dim * 8, norm_layer, use_dropout=use_dropout, use_bias=use_bias)

        self.up1 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 3, padding=1, stride=2, bias=use_bias))
        self.upnorm1 = norm_layer(conv_dim * 4, num_classes)
        self.up2 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 3, padding=1, stride=2, bias=use_bias))
        self.upnorm2 = norm_layer(conv_dim * 2, num_classes)
        self.up3 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 2, conv_dim * 1, 3, padding=p, stride=2, bias=use_bias))
        self.upnorm3 = norm_layer(conv_dim * 1, num_classes)

        self.c2 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 1, in_channels, 7, bias=use_bias))

    def forward(self, x, y):
        p2d = (3, 3, 3, 3)

        x_ = F.pad(x, p2d, 'reflect')
        x_ = self.norm1(self.c1(x_), y)
        x_ = F.relu(x_)

        x_ = self.downnorm1(self.down1(x_), y)
        x = F.relu(x_)
        x_ = self.downnorm2(self.down2(x_), y)
        x = F.relu(x_)
        x_ = self.downnorm3(self.down3(x_), y)
        x = F.relu(x_)

        x_ = self.res1(x, y)
        x_ = self.res2(x, y)
        x_ = self.res3(x, y)
        x_ = self.res4(x, y)

        x_ = self.upnorm1(self.up1(x_), y)
        x_ = F.relu(x_)
        x_ = self.upnorm2(self.up2(x_), y)
        X_ = F.relu(x_)
        x_ = self.upnorm3(self.up3(x_), y)
        X_ = F.relu(x_)

        x_ = F.pad(x_, p2d, 'reflect')
        x_ = self.norm2(self.c2(x_), y)
        x_ = torch.tanh(x_)

        return x_

class ResnetBlock(nn.Module):
    '''
    The code is adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    def __init__(self, conv_dim, norm_layer, padding_type='reflect', num_classes=10, use_dropout=False, use_bias=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(conv_dim, norm_layer, padding_type, num_classes, use_dropout, use_bias)

    def build_conv_block(self, conv_dim, norm_layer, padding_type, num_classes, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            conv_dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        self.padding_type = padding_type
        self.use_dropout = use_dropout
        p = 0
        if padding_type == 'zero':
            p = 1
        self.c1 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=p, bias=use_bias)
        self.norm1 = norm_layer(conv_dim, num_classes)

        self.c2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=p, bias=use_bias)
        self.norm2 = norm_layer(conv_dim, num_classes)

    def forward(self, x):
        """Forward function (with skip connections)"""
        p2d = (1, 1, 1, 1)
        if self.padding_type != 'zero':
            x_ = F.padding(x, p2d, self.padding_type)
            x_ = self.norm1(self.c1(x_), y)
            x_ = F.relu(x_)

            x_ = F.padding(x_, p2d, self.padding_type)
            x_ = self.norm2(self.c2(x_), y)
        else:
            x_ = self.norm1(self.c1(x_), y)
            x_ = F.relu(x_)

            x_ = self.norm2(self.c2(x_), y)
        out = x + x_  # add skip connections
        return out

class UnetGenerator(nn.Module):
    '''
    Fully convolutional generator for MNIST dataset
    U-Net architecture, encoder-decoder architecture with skip connections
    The code is adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    def __init__(self, in_channels=1, norm_layer=utils.ConditionalBatchNorm2d, num_classes=10, conv_dim=64, MNIST=True):
        super(UnetGenerator, self).__init__()
        self.u_block = UnetBlock(conv_dim * 8, conv_dim * 8, norm_layer, innermost=True, use_dropout=True)
        self.u_block = UnetBlock(conv_dim * 8, conv_dim * 8, norm_layer, submodule=self.u_block, use_dropout=True)
        self.u_block = UnetBlock(conv_dim * 4, conv_dim * 8, norm_layer, submodule=self.u_block, use_dropout=True)
        self.u_block = UnetBlock(conv_dim * 2, conv_dim * 4, norm_layer, submodule=self.u_block)
        self.u_block = UnetBlock(conv_dim * 1, conv_dim * 2, norm_layer, submodule=self.u_block)
        self.u_block = UnetBlock(in_channels , conv_dim * 1, norm_layer, submodule=self.u_block, outermost=True, MNIST=MNIST)

    def forward(self, x, y):
        return self.u_block(x, y)

class UnetBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, norm_layer,outer_channels=None, submodule=None, num_classes=10,
                 outermost=False, innermost=False, use_dropout=False, MNIST=True):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule
        self.use_dropout = use_dropout
        if outer_channels is None:
            outer_channels = in_channels
        if outermost:
            if MNIST:
                padding = 3
            else:
                padding = 1
            self.down = nn.utils.spectral_norm(nn.Conv2d(in_channels, inner_channels, 4, padding=padding, stride=2))
            self.up = nn.utils.spectral_norm(nn.ConvTranspose2d(inner_channels * 2, outer_channels, 4, padding=padding, stride=2))
        elif innermost:
            self.down = nn.utils.spectral_norm(nn.Conv2d(in_channels, inner_channels, 1))
            self.up = nn.utils.spectral_norm(nn.ConvTranspose2d(inner_channels, outer_channels, 1))
            self.downnorm = norm_layer(inner_channels, num_classes)
            self.upnorm = norm_layer(outer_channels, num_classes)
        else:
            self.down = nn.utils.spectral_norm(nn.Conv2d(in_channels, inner_channels, 4, padding=1, stride=2))
            self.up = nn.utils.spectral_norm(nn.ConvTranspose2d(inner_channels * 2, outer_channels, 4, padding=1, stride=2))
            self.downnorm = norm_layer(inner_channels, num_classes)
            self.upnorm = norm_layer(outer_channels, num_classes)

    def forward(self, x, y):
        if self.outermost:
            x_ = F.relu(self.down(x))
            x_ = self.submodule(x_, y)
            x_ = self.up(x_)
            return torch.tanh(x_)
        elif self.innermost:
            x_ = self.downnorm(self.down(x), y)
            x_ = F.relu(x_)
            x_ = self.upnorm(self.up(x_), y)
            if self.use_dropout:
                x_ = F.dropout2d(x)
            x_ = F.relu(x_)
            return torch.cat((x, x_), 1)
        else:
            x_ = self.downnorm(self.down(x), y)
            x_ = F.relu(x_)
            x_ = self.submodule(x_, y)
            x_ = self.upnorm(self.up(x_), y)
            if self.use_dropout:
                x_ = F.dropout2d(x)
            x_ = F.relu(x_)
            return torch.cat((x, x_), 1)

'''
class Generator(nn.Module):
    """
    Fully convolutional generator for MNIST dataset
    U-Net architecture, encoder-decoder architecture with skip connections
    """
    def __init__(self, in_channels=1, num_classes=10, conv_dim=64):
        super(Generator, self).__init__()

        # encoder
        self.e1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, conv_dim, 4, padding=3, stride=2))
        self.e2 = nn.utils.spectral_norm(nn.Conv2d(conv_dim, conv_dim * 2, 4, padding=1, stride=2))
        self.e3 = nn.utils.spectral_norm(nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, padding=1, stride=2))
        self.e4 = nn.utils.spectral_norm(nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, padding=1, stride=2))
        self.e5 = nn.utils.spectral_norm(nn.Conv2d(conv_dim * 8, conv_dim * 8, 4, padding=1, stride=2))
        self.e6 = nn.utils.spectral_norm(nn.Conv2d(conv_dim * 8, conv_dim * 8, 1))
        self.bn_e2 = selfattn.ConditionalBatchNorm2d(conv_dim * 2, num_classes)
        self.bn_e3 = selfattn.ConditionalBatchNorm2d(conv_dim * 4, num_classes)
        self.bn_e4 = selfattn.ConditionalBatchNorm2d(conv_dim * 8, num_classes)
        self.bn_e5 = selfattn.ConditionalBatchNorm2d(conv_dim * 8, num_classes)
        self.bn_e6 = selfattn.ConditionalBatchNorm2d(conv_dim * 8, num_classes)

        # decoder
        self.d1 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 8, conv_dim * 8, 1))
        self.d2 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 8 * 2, conv_dim * 8, 4, padding=1, stride=2))
        self.d3 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 8 * 2, conv_dim * 4, 4, padding=1, stride=2))
        self.d4 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 4 * 2, conv_dim * 2, 4, padding=1, stride=2))
        self.d5 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 2 * 2, conv_dim * 1, 4, padding=1, stride=2))
        self.d6 = nn.utils.spectral_norm(nn.ConvTranspose2d(conv_dim * 1 * 2, in_channels, 4, padding=3, stride=2))
        self.bn_d1 = selfattn.ConditionalBatchNorm2d(conv_dim * 8, num_classes)
        self.bn_d2 = selfattn.ConditionalBatchNorm2d(conv_dim * 8, num_classes)
        self.bn_d3 = selfattn.ConditionalBatchNorm2d(conv_dim * 4, num_classes)
        self.bn_d4 = selfattn.ConditionalBatchNorm2d(conv_dim * 2, num_classes)
        self.bn_d5 = selfattn.ConditionalBatchNorm2d(conv_dim * 1, num_classes)
a
    def forward(self, x, y):
        # input size: N, C, 28, 28
        x_ = F.relu(self.e1(x))
        e1 = x_.clone()
        # input size: N, conv_dim, 16, 16
        x_ = self.bn_e2(self.e2(x_), y)
        x_ = F.relu(x_)
        e2 = x_.clone()
        # input size: N, conv_dim * 2, 8, 8
        x_ = self.bn_e3(self.e3(x_), y)
        x_ = F.relu(x_)
        e3 = x_.clone()
        # input size: N, conv_dim * 4, 4, 4
        x_ = self.bn_e4(self.e4(x_), y)
        x_ = F.relu(x_)
        e4 = x_.clone()
        # input size: N, conv_dim * 8, 2, 2
        x_ = self.bn_e5(self.e5(x_), y)
        x_ = F.relu(x_)
        e5 = x_.clone()
        # input size: N, conv_dim * 8, 1, 1
        x_ = self.bn_e6(self.e6(x_), y)
        x_ = F.relu(x_)

        # input size: N, conv_dim * 8, 1, 1
        x_ = F.dropout2d(self.bn_d1(self.d1(x_), y))
        x_ = F.relu(x_)
        x_ = torch.cat((x_, e5), 1)
        # input size: N, conv_dim * 8 * 2, 2, 2
        x_ = F.dropout2d(self.bn_d2(self.d2(x_), y))
        x_ = F.relu(x_)
        x_ = torch.cat((x_, e4), 1)
        # input size: N, conv_dim * (8 + 4), 4, 4
        x_ = self.bn_d3(self.d3(x_), y)
        x_ = F.relu(x_)
        x_ = torch.cat((x_, e3), 1)
        # input size: N, conv_dim * (4 + 2), 8, 8
        x_ = self.bn_d4(self.d4(x_), y)
        x_ = F.relu(x_)
        x_ = torch.cat((x_, e2), 1)
        # input size: N, conv_dim * (2 + 1), 16, 16
        x_ = self.bn_d5(self.d5(x_), y)
        x_ = F.relu(x_)
        x_ = torch.cat((x_, e1), 1)
        # input size: N, conv_dim * 1 * 2, 28, 28
        x_ = self.d6(x_)
        x_ = torch.tanh(x_)

        return x_
'''
