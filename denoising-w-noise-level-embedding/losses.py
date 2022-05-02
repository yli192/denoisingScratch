import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        #a = torch.sqrt((diff * diff) + (self.eps*self.eps))
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        #print(loss.shape)

        return loss

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

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


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False),
                                  nn.Conv3d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats,in_channels=1):
        super(Encoder, self).__init__()
        self.conv = conv(in_channels,n_feat,3,1,1)
        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)


    def forward(self, x):
        x = self.conv(x)
        enc1 = self.encoder_level1(x)

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)

        return [enc1, enc2, enc3]

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Conv3d') != -1:
        # get the number of the inputs
        n_l = m.kernel_size[0]* m.kernel_size[0] *  m.in_channels
        std = np.sqrt(2/n_l)
        m.weight.data.normal_(0, std)
        #m.bias.data.fill_(0)

    # create a new model with these weights
Encoder3D = Encoder(n_feat=96, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=48)
Encoder3D.apply(weights_init_uniform_rule)
# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
# Encoder3D.fc3.register_forward_hook(get_activation('fc3'))
# output = model(x)
# activation['fc3']
#print(Encoder3D)
class GenericPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(GenericPerceptualLoss, self).__init__()
        self.network = Encoder3D.cuda()
        for bl in self.network.children():
            for p in bl.parameters():
                p.requires_grad = False
        #self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize


    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):

        loss = 0.0
        x = input
        y = target
        #for i, block in enumerate(self.network.children()):
        feas_x = self.network(x)
        feas_y = self.network(y)

        for i in range(len(feas_y)):
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(feas_x[i], feas_y[i])
        # if i in style_layers:
        #     act_x = x.reshape(x.shape[0], x.shape[1], -1)
        #     act_y = y.reshape(y.shape[0], y.shape[1], -1)
        #     gram_x = act_x @ act_x.permute(0, 2, 1)
        #     gram_y = act_y @ act_y.permute(0, 2, 1)
        #     loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

#GT = torch.rand(((2, 1, 32, 32, 32))).cuda()
#estimated = torch.rand(((2, 1, 32, 32, 32))).cuda()
#Loss = GenericPerceptualLoss()

#print(Loss(estimated, GT))


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class PoissonLikelihood_loss(nn.Module):
    def __init__(self, max_val=0):
        '''
        Poisson Likelihood loss function
        Email: jchen245@jhmi.edu
        Date: 02/21/2021
        :param max_val: the maximum value of the target.
        '''
        super(PoissonLikelihood_loss, self).__init__()
        self.max_val = max_val

    def forward(self, y_pred, y_true):
        eps = 1e-6
        #print(y_pred.shape,y_pred.view(y_pred.shape[0], -1).shape)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_true.shape[0], -1)

        p_l = -y_true+y_pred*torch.log(y_true + eps)-torch.lgamma(y_pred+1)

        return -torch.mean(p_l)

class Weighted_CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(Weighted_CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, weights):
        diff = x - y
        batch = x.shape[0]
        #a = weights.view(batch, -1, 1, 1, 1)
        #print("a",a)
        #scale the difference image with weights
        diff = diff * 10 * weights.view(batch, -1, 1, 1, 1)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,5,1,1)
        #print("the kernel shape:", self.kernel.shape)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh, kd = self.kernel.shape
        #print (kw,kh,kd)
        #print(img.size())
        img = F.pad(img, (kw//2, kh//2, kd//2, kw//2, kh//2, kd//2), mode='replicate')
        #print(img.size())
        return F.conv3d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class Weighted_EdgeLoss(nn.Module):
    def __init__(self):
        super(Weighted_EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,5,1,1)
        #print("the kernel shape:", self.kernel.shape)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = Weighted_CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh, kd = self.kernel.shape
        #print (kw,kh,kd)
        #print(img.size())
        img = F.pad(img, (kw//2, kh//2, kd//2, kw//2, kh//2, kd//2), mode='replicate')
        #print(img.size())
        return F.conv3d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y,weights):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y),weights)
        return loss
