"""Building 2D/3D U-Net."""

__author__ = 'Gary Y. Li'
import torch
from nets.conv_blocks import *
import numpy as np


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

def get_layers_config(input_dim=3, dropout_p=None, nonlin="LeakyReLU", norm_type="BatchNorm"):
    """
    returns a dictionary that stores pointers to the specific layers that will be used for conv, nonlin and norm operations throughtout the network
    :return: props
    """
    props = {}
    if input_dim == 2:
        props['conv_layer'] = nn.Conv2d
        props['dropout_layer'] = nn.Dropout2d
    elif input_dim == 3:
        props['conv_layer'] = nn.Conv3d
        props['dropout_layer'] = nn.Dropout3d
    else:
        raise NotImplementedError

    if norm_type == "BatchNorm":
        if input_dim == 2:
            props['norm_layer'] = nn.BatchNorm2d
        elif input_dim == 3:
            props['norm_layer'] = nn.BatchNorm3d
        props['norm_layer_kwargs'] = {'eps': 1e-5, 'affine': True}
    elif norm_type == "InstanceNorm":
        if input_dim == 2:
            props['norm_layer'] = nn.InstanceNorm2d
        elif input_dim == 3:
            props['norm_layer'] = nn.InstanceNorm3d
        props['norm_layer_kwargs'] = {'eps': 1e-5, 'affine': True}
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_layer'] = None
        props['dropout_layer_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_layer_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_layer_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': True}  # kernel size will be set by network!

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    else:
        raise ValueError

    return props


class Encoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_depth, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, layers_config, default_return_skips=True,
                 max_num_features=480):
        """
        :param input_channels: number of channels for the input to the convolutional layer
        :param base_num_features: number of filters in the first conv layer
        :param num_blocks_per_depth: a list or tuple of integers; each int is the number of conv layers in each stage
        :param feat_map_mul_on_downscale: out_num_features = base_num_features * (feat_map_mul_on_downscale)^stage
        :param pool_op_kernel_sizes: a tuple of pooling kernels used in each stage
        :param conv_kernel_sizes: a tuple of conv. kernels used in each stage, i.e., ((3,3,3),(3,3,3)) for conv3d
        :param layers_config: layers' configurations
        """
        super(Encoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.layers_config = layers_config

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_layer_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes) #number of statges is equal to the depth of the network

        if not isinstance(num_blocks_per_depth, (list, tuple)):
            num_blocks_per_depth = [num_blocks_per_depth] * num_stages
        else:
            assert len(num_blocks_per_depth) == num_stages

        self.num_blocks_per_depth = num_blocks_per_depth  # decoder may need this

        current_input_features = input_channels #for input, the current_input_features = channel_num
        for stage in range(num_stages):
            current_output_features = min(int(base_num_features * feat_map_mul_on_downscale ** stage), max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = composite_layer(current_input_features, current_output_features, current_kernel_size,
                                              layers_config, num_blocks_per_depth[stage])

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_layer_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_output_features

    def forward(self, x, return_skips=None):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        for s in self.stages: #to each stage (bunch of operations) in stages pass x through it; here the x is a set of feature maps
            x = s(x)
            if self.default_return_skips: #if this is true, then one input will generate multiple outputs
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

class Decoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_depth=None, layers_config=None, deep_supervision=False,
                 upscale_logits=False):
        super(Decoder, self).__init__()
        self.num_blocks_per_depth = num_blocks_per_depth
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.previous_stages = previous.stages
        self.previous_stage_output_features = previous.stage_output_features
        self.previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        self.previous_stage_conv_layer_kernel_size = previous.stage_conv_layer_kernel_size

        if layers_config is None:
            self.layers_config = previous.layers_config
        else:
            self.layers_config = layers_config

        if self.layers_config['conv_layer'] == nn.Conv2d:
            self.transpconv = nn.ConvTranspose2d
            self.upsample_mode = "bilinear"
        elif self.layers_config['conv_layer'] == nn.Conv3d:
            self.transpconv = nn.ConvTranspose3d
            self.upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv layer: %s" % str(self.props['conv_layer']))

        if self.num_blocks_per_depth is None:
            self.num_blocks_per_depth = previous.num_blocks_per_depth[:-1][::-1]

        assert len(self.num_blocks_per_depth) == len(previous.num_blocks_per_depth) - 1 #here the last bottleneck layer belongs to the encoder

        self.stage_pool_kernel_size = self.previous_stage_pool_kernel_size
        self.stage_output_features = self.previous_stage_output_features
        self.stage_conv_layer_kernel_size = self.previous_stage_conv_layer_kernel_size

        self.depth = len(self.previous_stages) - 1  # we have one less as the first depth here is what comes after the
        # bottleneck

        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        # only used for upsample_logits
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)

        for i, s in enumerate(np.arange(self.depth)[::-1]):
            num_features_below = self.previous_stage_output_features[s + 1]
            num_features_skip = self.previous_stage_output_features[s]

            self.tus.append(self.transpconv(num_features_below, num_features_skip, self.previous_stage_pool_kernel_size[s + 1],
                                       self.previous_stage_pool_kernel_size[s + 1], bias=False))
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(composite_layer(2 * num_features_skip, num_features_skip,
                                                 self.previous_stage_conv_layer_kernel_size[s], self.layers_config,
                                                 self.num_blocks_per_depth[i]))

            if deep_supervision and s != 0:
                seg_layer = self.props['conv_layer'](num_features_skip, num_classes, 1, 1, 0, 1, 1, False)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=self.upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = self.layers_config['conv_layer'](num_features_skip, num_classes, kernel_size= 1, stride=1, padding=0, dilation =1, groups=1, bias=False)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips, gt=None, loss=None):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]  # this is the bottleneck

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision and (i != len(self.tus) - 1):
                tmp = self.deep_supervision_outputs[i](x)
                if gt is not None:
                    tmp = loss(tmp, gt)
                seg_outputs.append(tmp)

        segmentation = self.segmentation_output(x)

        if self.deep_supervision:
            tmp = segmentation
            if gt is not None:
                tmp = loss(tmp, gt)
            seg_outputs.append(tmp)
            return seg_outputs[::-1]  # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation

class UNet(nn.Module):

    def __init__(self, input_channels, base_num_features, num_blocks_per_depth_encoder, feat_map_mul_on_downscale,
                 pool_layer_kernel_sizes, conv_layer_kernel_sizes, layers_config, num_classes, num_blocks_per_depth_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.base_num_features = base_num_features
        self.num_blocks_per_depth_encoder = num_blocks_per_depth_encoder
        self.feat_map_mul_on_downscale = feat_map_mul_on_downscale
        self.pool_layer_kernel_sizes = pool_layer_kernel_sizes
        self.conv_layer_kernel_sizes = conv_layer_kernel_sizes
        self.layers_config = layers_config
        self.num_classes = num_classes
        self.num_blocks_per_depth_decoder= num_blocks_per_depth_decoder
        #self.conv_layer = layers_config['conv_layer']


        self.encoder = Encoder(self.input_channels, self.base_num_features, self.num_blocks_per_depth_encoder,
                                            self.feat_map_mul_on_downscale, self.pool_layer_kernel_sizes, self.conv_layer_kernel_sizes,
                                            self.layers_config, default_return_skips=True, max_num_features=max_features)
        self.decoder = Decoder(self.encoder, self.num_classes, self.num_blocks_per_depth_decoder, self.layers_config,
                                            deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
