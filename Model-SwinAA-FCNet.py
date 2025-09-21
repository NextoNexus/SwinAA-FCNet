import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np

from torchinfo import summary


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class upConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(upConv, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class Decoder(nn.Module):
    def __init__(self, level, enc_nf):
        super(Decoder, self).__init__()

        self.corr = CorrTorch()

        self.upconv = upConv(3 * enc_nf[level] + 27, enc_nf[level - 1], 4, 2)
        if level == 4:
            self.upconv = upConv(enc_nf[level], enc_nf[level - 1], 4, 2)

        self.cconv = nn.Sequential(
            ConvInsBlock(3 * enc_nf[level - 1] + 27, 3 * enc_nf[level - 1] + 27, 3, 1),
            ConvInsBlock(3 * enc_nf[level - 1] + 27, 3 * enc_nf[level - 1] + 27, 3, 1)
        )
        '''self.res_conv = nn.Sequential(
            ConvInsBlock(3 * enc_nf[level - 1] + 27, 3 * enc_nf[level - 1] + 27, 3, 1),
            ConvInsBlock(3 * enc_nf[level - 1] + 27, 3 * enc_nf[level - 1] + 27, 3, 1)
        )'''

        self.output = nn.Conv3d(3 * enc_nf[level - 1] + 27, 3, 3, 1, 1)
        self.output.weight = nn.Parameter(Normal(0, 1e-5).sample(self.output.weight.shape))
        self.output.bias = nn.Parameter(torch.zeros(self.output.bias.shape))

    def forward(self, src, tgt, C):
        C = self.upconv(C)
        cost_volume = self.corr(tgt, src)
        four_input = torch.cat((tgt, src, cost_volume, C), dim=1)
        C = self.cconv(four_input)
        #C = C + self.res_conv(C)

        out = self.output(C)
        return C, out

class WindowChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons, window_size):
        super(WindowChannelAttention, self).__init__()

        self.window_size = window_size
        groups = input_channels // self.window_size
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             groups=groups, bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             groups=groups, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool3d(inputs, output_size=(1, 1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1, 1)
        return x * inputs


class WindowSpatialAttention(nn.Module):

    def __init__(self, in_channels, window_size):
        super().__init__()
        groups=in_channels // window_size
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.conv_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, groups=groups)

    def forward(self, inputs):
        inputs = self.conv(inputs) * inputs
        return self.conv_proj(inputs)


class SwinCSABlock(nn.Module):

    def __init__(self, in_channels, channelAttention_reduce=4, window_size=16, shift_size=8, alpha=0.1):
        super().__init__()

        groups = in_channels // window_size

        self.window_size=window_size
        self.shift_size=shift_size

        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.relu1=nn.LeakyReLU(alpha)
        self.ca = WindowChannelAttention(input_channels=in_channels,
                                         internal_neurons=in_channels // channelAttention_reduce,window_size=window_size)
        self.sa = WindowSpatialAttention(in_channels=in_channels,window_size=window_size)

        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(alpha)
        self.ffn = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, groups=groups)

    def forward(self, x):
        shortcut=x

        x = self.relu1(self.norm1(x))

        if self.shift_size > 0:
            shifted_x = torch.roll(x,-self.shift_size,dims=1)
        else:
            shifted_x = x

        shifted_x=self.sa(self.ca(shifted_x))

        if self.shift_size > 0:
            x = torch.roll(shifted_x, self.shift_size, dims=1)
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.ffn(self.relu2(self.norm2(x)))
        return x

class SwinCSA(nn.Module):

    def __init__(self, in_channels, channelAttention_reduce=4, window_size=16, shift_size=8, alpha=0.1):
        super().__init__()

        assert in_channels % window_size == 0, 'The number of channels must be divisible by the window size..'

        self.WCSA = SwinCSABlock(in_channels=in_channels,channelAttention_reduce=channelAttention_reduce,
                               window_size=window_size,shift_size=-1,alpha=alpha)
        self.SWCSA = SwinCSABlock(in_channels=in_channels, channelAttention_reduce=channelAttention_reduce,
                                 window_size=window_size,shift_size=shift_size, alpha=alpha)

        self.norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.SWCSA(self.WCSA(x))
        return self.act(self.norm(x))

class SE(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SE, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        #x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool3d(inputs, output_size=(1, 1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        #x2 = torch.sigmoid(x2)
        x = torch.sigmoid(x1 + x2)
        x = x.view(-1, self.input_channels, 1, 1, 1)

        out = x*inputs
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut=x
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        x = self.conv1(x)  # 通过卷积层得到注意力权重:(B,2,H,W)-->(B,1,H,W)
        attn = self.sigmoid(x)
        out = shortcut * attn
        return out


class CBAM(nn.Module):
    def __init__(self, in_channel, intermediate_channel, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = SE(input_channels=in_channel, internal_neurons=intermediate_channel)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.ca(x)  # 通过通道注意力机制得到的特征图,x:(B,C,H,W),ca(x):(B,C,1,1),out:(B,C,H,W)
        result = self.sa(out)  # 通过空间注意力机制得到的特征图,out:(B,C,H,W),sa(out):(B,1,H,W),result:(B,C,H,W)
        return result

class Encoder_1(nn.Module):
    def __init__(self,img_size, in_channel=1, enc_nf=None):
        super(Encoder_1, self).__init__()

        self.enc_nf = enc_nf
        self.H, self.W, self.D = img_size

        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, self.enc_nf[0], 3, 1),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.enc_nf[0], self.enc_nf[1], kernel_size=3, stride=2, padding=1),  # 80
            SwinCSA(in_channels=self.enc_nf[1], channelAttention_reduce=4, window_size=16, shift_size=8, alpha=0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.enc_nf[1], self.enc_nf[2], kernel_size=3, stride=2, padding=1),  # 40
            SwinCSA(in_channels=self.enc_nf[2], channelAttention_reduce=4, window_size=16, shift_size=8, alpha=0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(self.enc_nf[2], self.enc_nf[3], kernel_size=3, stride=2, padding=1),  # 20
            SwinCSA(in_channels=self.enc_nf[3], channelAttention_reduce=4, window_size=16, shift_size=8, alpha=0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(self.enc_nf[3], self.enc_nf[4], kernel_size=3, stride=2, padding=1),  # 20
            SwinCSA(in_channels=self.enc_nf[4], channelAttention_reduce=4, window_size=16, shift_size=8, alpha=0.1)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return [out0, out1, out2, out3, out4]

class Encoder_1_SE(nn.Module):
    def __init__(self,img_size, in_channel=1, enc_nf=None):
        super(Encoder_1_SE, self).__init__()

        self.enc_nf = enc_nf
        self.H, self.W, self.D = img_size

        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, self.enc_nf[0], 3, 1),
        )

        self.conv1 = nn.Sequential(
            ConvInsBlock(self.enc_nf[0], self.enc_nf[1], kernal_size=3, stride=2, padding=1),  # 80
            SE(input_channels=self.enc_nf[1], internal_neurons=self.enc_nf[1]//4)
        )

        self.conv2 = nn.Sequential(
            ConvInsBlock(self.enc_nf[1], self.enc_nf[2], kernal_size=3, stride=2, padding=1),  # 40
            SE(input_channels=self.enc_nf[2], internal_neurons=self.enc_nf[2] // 4)
        )

        self.conv3 = nn.Sequential(
            ConvInsBlock(self.enc_nf[2], self.enc_nf[3], kernal_size=3, stride=2, padding=1),  # 20
            SE(input_channels=self.enc_nf[3], internal_neurons=self.enc_nf[3] // 4)
        )

        self.conv4 = nn.Sequential(
            ConvInsBlock(self.enc_nf[3], self.enc_nf[4], kernal_size=3, stride=2, padding=1),  # 20
            SE(input_channels=self.enc_nf[4], internal_neurons=self.enc_nf[4] // 4)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return [out0, out1, out2, out3, out4]

class Encoder_1_CBAM(nn.Module):
    def __init__(self,img_size, in_channel=1, enc_nf=None):
        super(Encoder_1_CBAM, self).__init__()

        self.enc_nf = enc_nf
        self.H, self.W, self.D = img_size

        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, self.enc_nf[0], 3, 1),
        )

        self.conv1 = nn.Sequential(
            ConvInsBlock(self.enc_nf[0], self.enc_nf[1], kernal_size=3, stride=2, padding=1),  # 80
            CBAM(in_channel=self.enc_nf[1],intermediate_channel=self.enc_nf[1]//4,kernel_size=3)
        )

        self.conv2 = nn.Sequential(
            ConvInsBlock(self.enc_nf[1], self.enc_nf[2], kernal_size=3, stride=2, padding=1),  # 40
            CBAM(in_channel=self.enc_nf[2],intermediate_channel=self.enc_nf[2]//4,kernel_size=3)
        )

        self.conv3 = nn.Sequential(
            ConvInsBlock(self.enc_nf[2], self.enc_nf[3], kernal_size=3, stride=2, padding=1),  # 20
            CBAM(in_channel=self.enc_nf[3],intermediate_channel=self.enc_nf[3]//4,kernel_size=3)
        )

        self.conv4 = nn.Sequential(
            ConvInsBlock(self.enc_nf[3], self.enc_nf[4], kernal_size=3, stride=2, padding=1),  # 20
            CBAM(in_channel=self.enc_nf[4],intermediate_channel=self.enc_nf[4]//4,kernel_size=3)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return [out0, out1, out2, out3, out4]


class Encoder_2(nn.Module):
    def __init__(self, in_channel=2, parallel_sizes=None, enc_nf=None):
        super(Encoder_2, self).__init__()
        self.enc_nf = enc_nf
        self.parallel_sizes = parallel_sizes
        # Encoder functions
        self.encoder1 = ConvInsBlock(in_channel, self.enc_nf[0])  # 1
        self.initial = ConvInsBlock(self.enc_nf[0], self.enc_nf[0])

        self.encoder2 = ConvInsBlock(self.enc_nf[0], self.enc_nf[1], stride=2)  # 1 / 2
        self.LKG_2 = LargeGhostBlock(self.enc_nf[1], self.enc_nf[1], parallel_sizes=self.parallel_sizes)

        self.encoder3 = ConvInsBlock(self.enc_nf[1], self.enc_nf[2], stride=2)  # 1 / 4
        self.LKG_3 = LargeGhostBlock(self.enc_nf[2], self.enc_nf[2], parallel_sizes=self.parallel_sizes)

        self.encoder4 = ConvInsBlock(self.enc_nf[2], self.enc_nf[3], stride=2)  # 1 / 8
        self.LKG_4 = LargeGhostBlock(self.enc_nf[3], self.enc_nf[3], parallel_sizes=self.parallel_sizes)

        self.encoder5 = ConvInsBlock(self.enc_nf[3], self.enc_nf[4], stride=2)  # 1 / 16
        self.LKG_5 = LargeGhostBlock(self.enc_nf[4], self.enc_nf[4], parallel_sizes=self.parallel_sizes)

    def forward(self, x, y):
        # Get encoder activations
        input = torch.cat([x, y], dim=1)
        x0 = self.initial(self.encoder1(input))
        x1 = self.LKG_2(self.encoder2(x0))
        x2 = self.LKG_3(self.encoder3(x1))
        x3 = self.LKG_4(self.encoder4(x2))
        x4 = self.LKG_5(self.encoder5(x3))

        return [x0, x1, x2, x3, x4]


class LargeGhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, parallel_sizes, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # self.primary_conv = ConvBlock(in_channels,init_channels,batchnorm = True)
        self.primary_conv = ConvInsBlock(in_channels=in_channels, out_channels=init_channels)
        assert init_channels == new_channels, 'mid channel need == out channel in LargeGhostBlock!'
        self.cheap_operation_1 = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, parallel_sizes[0], 1, parallel_sizes[0] // 2, groups=init_channels),
            nn.InstanceNorm3d(new_channels),
            nn.LeakyReLU(0.1)
        )

        self.cheap_operation_2 = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, parallel_sizes[1], 1, parallel_sizes[1] // 2, groups=init_channels),
            nn.InstanceNorm3d(new_channels),
            nn.LeakyReLU(0.1)
        )
        self.cheap_operation_3 = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, parallel_sizes[2], 1, parallel_sizes[2] // 2, groups=init_channels),
            nn.InstanceNorm3d(new_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        init_feature = self.primary_conv(x)
        ghost_feature_1 = self.cheap_operation_1(init_feature)
        ghost_feature_2 = self.cheap_operation_2(init_feature)
        ghost_feature_3 = self.cheap_operation_3(init_feature)
        fused_feature = init_feature + ghost_feature_1 + ghost_feature_2 + ghost_feature_3
        out = torch.cat([init_feature, fused_feature], dim=1)
        return out[:, :self.out_channels, :, :, :]

class Fourier_down(nn.Module):

    def __init__(self,img_size):
        super().__init__()

        self.pooling=nn.MaxPool3d(2)
        self.H = img_size[0]
        self.W = img_size[1]
        self.D = img_size[2]


    def forward(self, x):
        vout_fft1 = torch.fft.fftshift(torch.fft.fftn(x))
        cropped = vout_fft1[:, :, self.H // 4:3 * self.H // 4, self.W // 4:3 * self.W // 4, self.D // 4:3 * self.D // 4]

        emptied = vout_fft1.clone()
        emptied[:, :, self.H // 4:3 * self.H // 4, self.W // 4:3 * self.W // 4, self.D // 4:3 * self.D // 4] = 0

        cropped = torch.real(torch.fft.ifftn(torch.fft.ifftshift(cropped)))
        emptied = torch.real(torch.fft.ifftn(torch.fft.ifftshift(emptied)))
        emptied = self.pooling(emptied)

        return cropped + emptied

class TeP(nn.Module):
    def __init__(self, start_channel, inshape=(80, 96, 112)):
        self.C = start_channel
        self.inshape = inshape
        self.enc_nf = [self.C, 2 * self.C, 4 * self.C, 8 * self.C, 16 * self.C]
        bias_opt = True

        super(TeP, self).__init__()
        self.Encoder_x = Encoder_1(img_size=inshape, in_channel=1, enc_nf=self.enc_nf)
        self.Encoder_y = Encoder_1(img_size=inshape, in_channel=1, enc_nf=self.enc_nf)
        # self.Encoder_xy=Encoder_2(in_channel=2,parallel_sizes=parallel_sizes,enc_nf=self.enc_nf)

        self.corr = CorrTorch()
        self.cconv5 = nn.Sequential(
            ConvInsBlock(2 * self.enc_nf[4] + 27, self.enc_nf[4], 3, 1),
            ConvInsBlock(self.enc_nf[4], self.enc_nf[4], 3, 1)
        )
        '''self.res_conv=nn.Sequential(
            ConvInsBlock(self.enc_nf[4], self.enc_nf[4], 3, 1),
            ConvInsBlock(self.enc_nf[4], self.enc_nf[4], 3, 1)
        )'''
        self.output5 = nn.Conv3d(self.enc_nf[4], 3, 3, 1, 1)
        self.output5.weight = nn.Parameter(Normal(0, 1e-5).sample(self.output5.weight.shape))
        self.output5.bias = nn.Parameter(torch.zeros(self.output5.bias.shape))

        self.decoder4 = Decoder(level=4, enc_nf=self.enc_nf)
        self.decoder3 = Decoder(level=3, enc_nf=self.enc_nf)
        self.decoder2 = Decoder(level=2, enc_nf=self.enc_nf)
        self.decoder1 = Decoder(level=1, enc_nf=self.enc_nf)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

    def fourier_recover(self, input, patch_size, recover_size):
        vout_1, vout_2, vout_3 = input[:, 0:1, :, :, :], input[:, 1:2, :, :, :], input[:, 2:3, :, :, :]
        vout_1 = vout_1.squeeze().squeeze()
        vout_2 = vout_2.squeeze().squeeze()
        vout_3 = vout_3.squeeze().squeeze()
        vout_fft1 = torch.fft.fftshift(torch.fft.fftn(vout_1))
        vout_fft2 = torch.fft.fftshift(torch.fft.fftn(vout_2))
        vout_fft3 = torch.fft.fftshift(torch.fft.fftn(vout_3))
        a = (recover_size[2] - patch_size[2]) // 2
        b = recover_size[2] - patch_size[2] - a

        c = (recover_size[1] - patch_size[1]) // 2
        d = recover_size[1] - patch_size[1] - c

        e = (recover_size[0] - patch_size[0]) // 2
        f = recover_size[0] - patch_size[0] - e
        p3d = (a, b, c, d, e, f)
        '''p3d = (
            (recover_size[2] - patch_size[2]) // 2, (recover_size[2] - patch_size[2]) // 2, (recover_size[1] - patch_size[1]) // 2,
            (recover_size[1] - patch_size[1]) // 2,
            (recover_size[0] - patch_size[0]) // 2, (recover_size[0] - patch_size[0]) // 2)'''
        vout_fft1 = F.pad(vout_fft1, p3d, "constant", 0)
        vout_fft2 = F.pad(vout_fft2, p3d, "constant", 0)
        vout_fft3 = F.pad(vout_fft3, p3d, "constant", 0)
        vdisp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_fft1)))  # * (img_x * img_y * img_z / 8))))
        vdisp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_fft2)))  # * (img_x * img_y * img_z / 8))))
        vdisp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_fft3)))  # * (img_x * img_y * img_z / 8))))
        D_f_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0),
                            vdisp_mf_3.unsqueeze(0).unsqueeze(0)], dim=1)
        return D_f_xy

    def forward(self, x, y):
        M1, M2, M3, M4, M5 = self.Encoder_x(x)
        F1, F2, F3, F4, F5 = self.Encoder_y(y)
        # features_xy = self.Encoder_xy(x,y)
        intermediate_flows = []

        H, W, D = self.inshape
        H5, W5, D5 = (H // 16, W // 16, D // 16)
        H4, W4, D4 = (H // 8, W // 8, D // 8)
        H3, W3, D3 = (H // 4, W // 4, D // 4)
        H2, W2, D2 = (H // 2, W // 2, D // 2)
        H1, W1, D1 = (H, W, D)

        correlation=self.corr(F5,M5)
        C5 = self.cconv5(torch.cat([F5, M5, correlation], dim=1))
        #C5 = C5 + self.res_conv(C5)
        output5 = self.output5(C5)
        flow54 = self.fourier_recover(output5, patch_size=(H5, W5, D5), recover_size=(H4, W4, D4))
        flow53 = self.fourier_recover(output5, patch_size=(H5, W5, D5), recover_size=(H3, W3, D3))
        flow52 = self.fourier_recover(output5, patch_size=(H5, W5, D5), recover_size=(H2, W2, D2))
        flow51 = self.fourier_recover(output5, patch_size=(H5, W5, D5), recover_size=(H1, W1, D1))
        flow54 = self.diff[3](flow54)
        flow53 = self.diff[2](flow53)
        flow52 = self.diff[1](flow52)
        flow51 = self.diff[0](flow51)

        fused_flow4 = flow54
        fused_flow3 = flow53
        fused_flow2 = flow52
        fused_flow1 = flow51

        flow = fused_flow4  # (10,12,24)
        intermediate_flows.append(flow)

        M4 = self.warp[3](M4, flow)
        C4, output4 = self.decoder4(M4, F4, C5)
        flow43 = self.fourier_recover(output4, patch_size=(H4, W4, D4), recover_size=(H3, W3, D3))  # (10,12,14)
        flow42 = self.fourier_recover(output4, patch_size=(H4, W4, D4), recover_size=(H2, W2, D2))
        flow41 = self.fourier_recover(output4, patch_size=(H4, W4, D4), recover_size=(H1, W1, D1))
        flow43 = self.diff[2](flow43)
        flow42 = self.diff[1](flow42)
        flow41 = self.diff[0](flow41)

        fused_flow3 = self.warp[2](fused_flow3, flow43) + flow43
        fused_flow2 = self.warp[1](fused_flow2, flow42) + flow42
        fused_flow1 = self.warp[0](fused_flow1, flow41) + flow41

        flow = fused_flow3
        intermediate_flows.append(flow)

        M3 = self.warp[2](M3, flow)
        C3, output3 = self.decoder3(M3, F3, C4)
        flow32 = self.fourier_recover(output3, patch_size=(H3, W3, D3), recover_size=(H2, W2, D2))
        flow31 = self.fourier_recover(output3, patch_size=(H3, W3, D3), recover_size=(H1, W1, D1))
        flow32 = self.diff[1](flow32)
        flow31 = self.diff[0](flow31)

        fused_flow2 = self.warp[1](fused_flow2, flow32) + flow32
        fused_flow1 = self.warp[0](fused_flow1, flow31) + flow31

        flow = fused_flow2
        intermediate_flows.append(flow)

        M2 = self.warp[1](M2, flow)
        C2, output2 = self.decoder2(M2, F2, C3)
        flow21 = self.fourier_recover(output2, patch_size=(H2, W2, D2), recover_size=(H1, W1, D1))
        flow21 = self.diff[0](flow21)

        fused_flow1 = self.warp[0](fused_flow1, flow21) + flow21

        flow = fused_flow1
        intermediate_flows.append(flow)

        M1 = self.warp[0](M1, flow)
        C1, output1 = self.decoder1(M1, F1, C2)
        flow11 = self.fourier_recover(output1, patch_size=(H1, W1, D1), recover_size=(H1, W1, D1))
        flow11 = self.diff[0](flow11)

        fused_flow1 = self.warp[0](fused_flow1, flow11) + flow11

        flow = fused_flow1
        intermediate_flows.append(flow)

        warped = self.warp[0](x, flow)

        return warped, flow, intermediate_flows


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class CorrTorch(nn.Module):
    def __init__(self, pad_size=1, kernel_size=1, max_displacement=1, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad3d(pad_size, 0)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsetz, offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1)])

        dep, hei, wid = in1.shape[2], in1.shape[3], in1.shape[4]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dz:dz + dep, dy:dy + hei, dx:dx + wid], 1, keepdim=True)
            for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))
        ], 1)
        return output


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() == True else 'cpu')
    '''Clue5 = Clue(level=2).to(device)
    x1 = torch.randn([1, 3, 80, 96, 112]).to(device)
    summary(Clue5, input_data=x1)
    output=Clue5(x1)
    print(output.shape)'''

    '''decoder=Decoder(level=4,enc_nf=[16,32,64,128,256]).to(device)
    x1 = torch.randn([1, 128, 16, 16, 16]).to(device)
    x2 = torch.randn([1, 128, 16, 16, 16]).to(device)
    x3 = torch.randn([1, 256, 8, 8, 8]).to(device)
    summary(decoder, input_data=(x1, x2,x3))
    C, out = decoder(x1, x2, x3)
    print(C.shape,out.shape)'''

    # x1 = torch.randn([1, 1, 128, 128, 128]).to(device)
    # x2 = torch.randn([1, 1, 128, 128, 128]).to(device)
    # encoder_scsa=Encoder_1(img_size=(128, 128, 128),in_channel=1,enc_nf=[16,32,64,128,256])
    # #model = Encoder_1(img_size=(80,96,80),in_channel=1,enc_nf=[8,16,32,64,128])
    # #model = TeP(start_channel=16, inshape=(128, 128, 128)).to(device)
    # summary(encoder_scsa, input_data=x1)
    #output = model(x1)
    #print(output.shape)
    x1 = torch.randn([1, 1, 128, 128, 128]).to(device)
    x2 = torch.randn([1, 1, 128, 128, 128]).to(device)
    model = TeP(start_channel=16, inshape=(128, 128, 128)).to(device)
    summary(model, input_data=(x1, x2))