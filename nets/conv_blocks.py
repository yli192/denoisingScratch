from copy import deepcopy
from torch import nn

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input

class local_convBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, layers_config):
        """A ConvDropoutNormReLU block
        :param input_channels: number of channels for the input to the convolutional layer
        :param output_channels: number of filter channels
        :param kernel_size: convolution kernal size. A tuple or list
        :param layers_config: a dict of network properties; the keys are string arguments and the corresponding items are functionals in nn.Modules like nn.Conv3d
        :return: a tensor passed through all the layers defined below
        """
        super(local_convBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.layers_config = layers_config

        self.conv = layers_config['conv_layer'](self.input_channels, self.output_channels, self.kernel_size,
                                             padding=[(i - 1) // 2 for i in self.kernel_size],
                                             **self.layers_config['conv_layer_kwargs'])  #conv_layer_kwargs: {'p': dropout_p, 'inplace': True}

        if self.layers_config['dropout_layer'] is not None:
            self.do = self.layers_config['dropout_layer'](**self.layers_config['dropout_layer_kwargs'])
        else:
            self.do = Identity()

        if self.layers_config['norm_layer'] is not None:
            self.norm = self.layers_config['norm_layer'](self.output_channels, **self.layers_config['norm_layer_kwargs'])
        else:
            self.norm = Identity()

        self.nonlin = self.layers_config['nonlin'](**self.layers_config['nonlin_kwargs'])

        self.all = nn.Sequential(self.conv, self.do, self.norm, self.nonlin)

    def forward(self,x):
        return self.all(x)


class composite_layer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, layers_config, num_convs_per_depth):
        """
        if layers_config['dropout_layer'] is None then no dropout
        if layers_config['norm_layer'] is None then no norm
        :param input_channels: number of channels for the input to the convolutional layer
        :param output_channels: number of filter channels
        :param kernel_size: convolution kernal size. A tuple or list
        :param layers_config:  a dict of network properties; the keys are string arguments and the corresponding items are functionals in nn.Modules like nn.Conv3d
        :parm num_comvs: num_blocks_per_stage in encoder or decoder
        """
        super(composite_layer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.layers_config = layers_config
        self.num_convs_per_depth = num_convs_per_depth

        self.composite_local_convBlocks = nn.Sequential(
            local_convBlock(self.input_channels, self.output_channels, self.kernel_size, self.layers_config),
            *[local_convBlock(self.output_channels, self.output_channels, self.kernel_size, self.layers_config) for _ in  #passing a list of ConvDropoutNormReLU operations/layers based on the num_convs
              range(self.num_convs_per_depth - 1)]
        )

    def forward(self, x):
        return self.composite_local_convBlocks(x)