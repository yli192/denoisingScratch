"""
## Author: Gary Y. Li
## Modified based on Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)
swish = F.silu

class split_layer(nn.Module):
    def __init__(self,n_channels,dim=1):
        super().__init__()
        self.n_channels=n_channels
        self.dim = dim

    def forward(self, input):
        return torch.split(input,self.n_channels,self.dim)


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin

@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

class NLEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim


    def forward(self, input):
        shape = input.shape #10,5
        sinusoid_in = nn.linear(input.shapep[-1],self.dim) #
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1) #
        pos_emb = pos_emb.view(*shape, self.dim) #4,320,80

        return pos_emb


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CALayer_wNLE(nn.Module):
    def __init__(self, channel, NLE_dim, reduction=16, bias=False):
        super(CALayer_wNLE, self).__init__()
        NLE_dim_out = channel
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveMaxPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x, noiseLevel_feat):
        batch = x.shape[0] #here the noiseLevel_feat is embedded and should be batch, noiseLevel_dim
        len = noiseLevel_feat.shape[1]
        # print(x.shape)
        beta = noiseLevel_feat[:, 0:int(len / 2)]  # has the dimension of n_feat
        gamma = noiseLevel_feat[:, int(len / 2):]
        y = self.avg_pool(x) #4,80,1,1,1
        y = self.conv_du(y)
        moduled_y = y * beta.view(batch, -1, 1, 1, 1) + gamma.view(batch, -1, 1, 1, 1)
        out = x * moduled_y
        return out

class FMB(nn.Module):
    #this layer modulates the input feature map to condition it on the input noise level feature vector
    def __init__(self, n_feat, NLE_dim_in):
        super(FMB, self).__init__()

        self.NLE_linear_beta = nn.Sequential(
            linear(NLE_dim_in, 2 * n_feat),
            Swish(),
            linear(2 * n_feat, n_feat)
        )
        self.NLE_linear_gamma = nn.Sequential(
            linear(NLE_dim_in, 2 * n_feat),
            Swish(),
            linear(2 * n_feat, n_feat)
        )
        #self.body = nn.Sequential(*modules_body)

    def forward(self, x, noiselevel_feat):
        batch = x.shape[0]
        beta = self.NLE_linear_beta(noiselevel_feat) # has the dimension of n_feat
        gamma = self.NLE_linear_gamma(noiselevel_feat)
        res = x * beta.view(batch, -1, 1, 1, 1) + gamma.view(batch, -1, 1, 1, 1)
        return res

class FMB_simple(nn.Module):
    #this layer modulates the input feature map to condition it on the input noise level feature vector
    def __init__(self):
        super(FMB_simple, self).__init__()

    def forward(self, x, noiselevel_feat):
        batch = x.shape[0]
        len = noiselevel_feat.shape[1]
        #print(x.shape)
        beta = noiselevel_feat[:,0:int(len/2)] # has the dimension of n_feat
        gamma = noiselevel_feat[:,int(len/2):]
        res = x * beta.view(batch, -1, 1, 1, 1) + gamma.view(batch, -1, 1, 1, 1)
        return res


##########################################################################
class CAB(nn.Module):
    def __init__(self, n_feat,  kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias) #n_feat = channel, noiseLevel_dim
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) #x.shape=[4,80,32,32,32] and res.shape=[4,80,32,32,32]
        res = self.CA(res)
        res += x
        return res

class CAB_wNLE(nn.Module):
    def __init__(self, n_feat,  kernel_size, NLE_dim, reduction, bias, act):
        super(CAB_wNLE, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.body = nn.Sequential(*modules_body)
        self.CA_wNLE = CALayer_wNLE(n_feat, NLE_dim, reduction, bias=bias) #n_feat = channel, noiseLevel_dim


    def forward(self, x, noiseLevel_feat):
        res = self.body(x)  # x.shape=[4,80,32,32,32] and res.shape=[4,80,32,32,32]
        res = self.CA_wNLE(res,noiseLevel_feat)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img




##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False),
                                  nn.Conv3d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class DownSample1D(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample1D, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5),
                                  nn.Conv1d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                                nn.Conv3d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                                nn.Conv3d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ORB_wNLE(nn.Module):
    def __init__(self, n_feat, kernel_size, NLE_dim, reduction, act, bias, num_cab):
        super(ORB_wNLE, self).__init__()
        #modules_body = []
        self.module1 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module2 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module3 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module4 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module5 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module6 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module7 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.module8 = CAB_wNLE(n_feat, kernel_size, NLE_dim, reduction, bias=bias, act=act)
        self.tail = conv(n_feat, n_feat, kernel_size)
        #self.body = nn.Sequential(modules_body)

    def forward(self, x, noiseLevel_feat):
        res = self.module1(x,noiseLevel_feat)
        res = self.module2(res,noiseLevel_feat)
        res = self.module3(res,noiseLevel_feat)
        res = self.module4(res,noiseLevel_feat)
        res = self.module5(res,noiseLevel_feat)
        res = self.module6(res,noiseLevel_feat)
        res = self.module7(res,noiseLevel_feat)
        res = self.module8(res,noiseLevel_feat)
        res = self.tail(res)
        res += x
        return res

class ORSNet_simple_wFMB(nn.Module):
    def __init__(self, n_feat, kernel_size, NLE_dim, reduction, act, bias, num_cab):
        super(ORSNet_simple_wFMB, self).__init__()

        self.orb1 = ORB_wNLE(n_feat, kernel_size, NLE_dim, reduction, act, bias, num_cab)
        self.orb2 = ORB_wNLE(n_feat, kernel_size, NLE_dim, reduction, act, bias, num_cab)
        self.orb3 = ORB_wNLE(n_feat, kernel_size, NLE_dim, reduction, act, bias, num_cab)

        #self.featMod = FMB_simple()
        self.activation = act


    def forward(self, x, noiseLevel_feat):

        x = self.orb1(x,noiseLevel_feat)
        x = self.orb2(x,noiseLevel_feat)
        x = self.orb3(x,noiseLevel_feat)

        return x

class ORSNet_simple(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet_simple, self).__init__()

        self.orb1 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)


    def forward(self, x):
        x = self.orb1(x)
        #x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        #x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        #x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x

##########################################################################


class ORSNet3D_wNLEEncDec_ORSOnly_OneMLP(nn.Module):
    def __init__(self, in_c=1, out_c=1, in_NLF=3, n_feat=32, num_cab=1,
                 kernel_size=3, reduction=4, bias=False):
        super(ORSNet3D_wNLEEncDec_ORSOnly_OneMLP, self).__init__()
        NLE_dim = 2 * n_feat
        act = nn.PReLU()

        self.NLE = nn.Sequential(
            nn.Linear(in_NLF, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, NLE_dim, bias=True)
        )

        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.stage3_orsnet_simple_wFMB = ORSNet_simple_wFMB(n_feat, kernel_size, NLE_dim, reduction, act, bias,
                                    num_cab)

        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)


    def forward(self, x3_img, noiseLevel_input):

        noiseLevel_feat = self.NLE(noiseLevel_input)
        # ## Compute Shallow Features
        x3 = self.shallow_feat3(x3_img)

        x3_cat = self.stage3_orsnet_simple_wFMB(x3,noiseLevel_feat)

        stage3_img = self.tail(x3_cat)

        return [stage3_img + x3_img]
