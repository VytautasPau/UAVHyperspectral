import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


class EncConvBlock(nn.Module):  # group norm, conv3d, activation
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=4, padding=1):
        super(EncConvBlock, self).__init__()
        if num_groups > 1 and in_channels % num_groups == 0:
            bias = False
            self.g1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            # self.g1 = nn.Identity()
        else:
            bias = True
            self.g1 = nn.Identity()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.g1(x)
        x = self.c(x)
        x = self.r(x)
        return x


class EncConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, num_groups=4, padding=1, drop=0.05):
        super(EncConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels * 2
        if in_channels == 1:  # for first convolution, groun norm = 1, due to input of  channel
            self.add_module("encConv1", EncConvBlock(in_channels, mid_channels, kernel_size=kernel_size, num_groups=1, padding=padding))
        else:
            self.add_module("encConv1", EncConvBlock(in_channels, mid_channels, kernel_size=kernel_size, num_groups=num_groups, padding=padding))
        self.add_module("encConv2", EncConvBlock(mid_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups, padding=padding))
        self.add_module("drop1", nn.Dropout(p=drop))


class DecConvBlockConcat(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, num_groups=4, padding=1):
        super(DecConvBlockConcat, self).__init__()
        if mid_channels is None:
            mid_channels = int((in_channels + out_channels) / 2)
        if num_groups > 1 and in_channels % num_groups == 0 and mid_channels % num_groups == 0:
            pass
        else:
            num_groups = 1
        self.conv = EncConvBlock(in_channels, mid_channels, kernel_size=kernel_size, num_groups=num_groups, padding=padding) # in_channels are after concat, su double the encoded value
        self.conv2 = EncConvBlock(mid_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups, padding=padding)

    def forward(self, encoded, x):
        output_size = encoded.size()[2:]
        # upsample the input and return
        # print(x.size())
        x = F.interpolate(x, size=output_size, mode="nearest")
        x = torch.cat((encoded, x), dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x


class DecConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, num_groups=4, padding=1):
        super(DecConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = int((in_channels + out_channels) / 2)
        if num_groups > 1 and in_channels % num_groups == 0 and mid_channels % num_groups == 0:
            pass
        else:
            num_groups = 1
        self.conv = EncConvBlock(in_channels, mid_channels, kernel_size=kernel_size, num_groups=num_groups, padding=padding) # in_channels are after concat, su double the encoded value
        self.conv2 = EncConvBlock(mid_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups, padding=padding)

    def forward(self, encoded, x):
        output_size = encoded.size()[2:]
        # upsample the input and return
        # print(x.size())
        x = F.interpolate(x, size=output_size, mode="nearest")
        # x = torch.cat((encoded, x), dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPool, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.c2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):  #  batch, wl, col, row
        x = torch.transpose(x, 1, 2)  # wl, col switch
        x = self.c1(x)
        x = torch.transpose(x, 1, 3)  # col, row switch
        x = self.c2(x)
        x = torch.transpose(x, 1, 2) # row wl switch
        return x
 

class UNet(nn.Module):
    def __init__(self, in_bands, out_class, col, row, pool_kernel_size=2, drop1=0.1, drop2=0.05):
        super().__init__()
        self.out_class = out_class
        self.drop = drop1
        self.drop_inner = drop2
        
        self.e1 = EncConv(in_bands, in_bands // 2, int(in_bands // 1.5), drop=self.drop_inner)
        self.em1 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        # self.em1 = ConvPool(col, col // 2)
        self.bn1 = nn.BatchNorm2d(in_bands // 2, momentum=0.5)
        self.oc1 = in_bands // 2  # out channel 1
        
        self.e2 = EncConv(self.oc1, self.oc1 // 2, int(self.oc1 // 1.5), drop=self.drop_inner)
        self.em2 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        # self.em2 = ConvPool(col // 2, col // 4)
        self.bn2 = nn.BatchNorm2d(self.oc1 // 2, momentum=0.5)
        self.oc2 = self.oc1 // 2 # out channel 2

        self.e3 = EncConv(self.oc2, self.oc2 // 2, int(self.oc2 // 1.5), drop=self.drop_inner)
        self.em3 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        # self.em3 = ConvPool(col // 4, col // 8)
        self.oc3 = self.oc2 // 2 # out channel 3

        self.conv4 = nn.Conv2d(self.oc3, self.oc3 // 2, kernel_size=3, padding=1, bias=True)  # half the channels to keep structure after concat (optional, testing only).
        self.func4 = nn.ReLU()

        # decode to endmembers
        self.decoder2 = nn.Sequential(
            nn.Conv2d(self.oc2, out_class, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LeakyReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(self.oc3, out_class, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LeakyReLU(),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(self.oc3 // 2, out_class, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LeakyReLU(),
        )
        self.flat = int(col // 8 ) * int(row // 8) + int(col // 4) * int(row // 4) + int(col // 2) * int(row // 2)
        self.gn = GaussianNoise(sigma=0.1, is_relative_detach=True)
        self.maxpool = nn.MaxPool1d(2)
        self.decoder = nn.Sequential(
            nn.Linear(self.flat // 2, in_bands),
            # nn.ReLU(),
            nn.Sigmoid(),
        )
        
        # with concat
        # self.d1 = DecConvBlockConcat(int(self.oc3 * 1.5), self.oc2)
        # self.d2 = DecConvBlockConcat(int(self.oc2 * 2), self.oc1)
        #self.d3 = DecConvBlockConcat(int(self.oc1 * 2), in_bands)
        #without concat
        self.d1 = DecConvBlock(int(self.oc3 // 2), self.oc2)
        self.d2 = DecConvBlock(int(self.oc2 ), self.oc1)
        self.d3 = DecConvBlock(int(self.oc1 ), in_bands)

        # abundance get
        self.conv5 = nn.Conv2d(in_bands, out_class, kernel_size=3, padding=1, bias=True)  # abundance matrix

        self.smooth = nn.Sequential(
            nn.Conv2d(out_class, out_class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )
        self.dr1 = nn.Dropout(self.drop)
        self.dr2 = nn.Dropout(self.drop)
        self.dr3 = nn.Dropout(self.drop)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.dr1(self.e1(x))
        x2 = self.e2(self.bn1(self.em1(x1)))
        x3 = self.e3(self.bn2(self.em2(x2)))
        x4 = self.dr2(self.em3(x3))
        x5 = self.func4(self.conv4(x4))
        endm2 = self.decoder2(x2)
        endm2 = torch.flatten(endm2, start_dim=2)
        endm3 = self.decoder3(x3)
        endm3 = torch.flatten(endm3, start_dim=2)
        endm5 = self.decoder5(x5)
        endm = torch.flatten(endm5, start_dim=2)
        endm = torch.cat((endm, endm2, endm3), dim=2)
        endm = self.gn(endm)
        endm = self.maxpool(endm)
        #endm = endm.permute(0, 2, 1)
        endm = self.decoder(endm).squeeze()

        x6 = self.d1(x3, x5)
        x6 = self.dr3(self.d2(x2, x6))
        x6 = self.d3(x1, x6)
        abd = self.smooth(self.conv5(x6))
        abd = abd.permute(0, 2, 3, 1)
        mult = torch.matmul(abd, endm)
        mult = mult.permute(0, 3, 1, 2)

        return abd, mult, endm
