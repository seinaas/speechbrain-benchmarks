import torch
import torch.nn as nn
import speechbrain as sb
from torch import Tensor

class ConvModule(nn.Module):
  """Module for performing temporal, spatial, and separable temporal convolution in sequence.

    Arguments
    ---------
    C: int
        Input channels.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_depth_multiplier: int
        Depth multiplier of the 2d spatial depthwise convolution.
    cnn_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution.
    cnn_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution.
    cnn_septemporal_depth_multiplier: int
        Depth multiplier of the 2d temporal separable convolution.
    cnn_septemporal_kernelsize: tuple
        Kernel size of the 2d temporal separable convolution.
    cnn_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    activation: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> conv_module = ConvModule(...)
    #>>> output = conv_module(inp_tensor)
    """

  i = 0
  def __init__(
    self,
    C,
    cnn_temporal_kernels,
    cnn_temporal_kernelsize,
    cnn_spatial_depth_multiplier,
    cnn_spatial_max_norm,
    cnn_spatial_pool,
    cnn_septemporal_depth_multiplier,
    cnn_septemporal_point_kernels,
    cnn_septemporal_kernelsize,
    cnn_septemporal_pool,
    cnn_pool_type,
    activation,
    dropout
  ):
    super().__init__()

    self.conv_module = torch.nn.Sequential()
    # Temporal convolution
    self.conv_module.add_module(
        f'conv_{ConvModule.i}_0',
        sb.nnet.CNN.Conv2d(
            in_channels=1,
            out_channels=cnn_temporal_kernels,
            kernel_size=cnn_temporal_kernelsize,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        ),
    )
    self.conv_module.add_module(
        f'bnorm_{ConvModule.i}_0',
        sb.nnet.normalization.BatchNorm2d(
            input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
        ),
    )

    # Spatial depthwise convolution
    cnn_spatial_kernels = (
        cnn_spatial_depth_multiplier * cnn_temporal_kernels
    )
    self.conv_module.add_module(
        f'conv_{ConvModule.i}_1',
        sb.nnet.CNN.Conv2d(
            in_channels=cnn_temporal_kernels,
            out_channels=cnn_spatial_kernels,
            kernel_size=(1, C),
            groups=cnn_temporal_kernels,
            padding="valid",
            bias=False,
            max_norm=cnn_spatial_max_norm,
            swap=True,
        ),
    )
    self.conv_module.add_module(
        f'bnorm_{ConvModule.i}_1',
        sb.nnet.normalization.BatchNorm2d(
            input_size=cnn_spatial_kernels, momentum=0.1, affine=True,
        ),
    )
    self.conv_module.add_module(f'act_{ConvModule.i}_1', activation)
    self.conv_module.add_module(
        f'pool_{ConvModule.i}_1',
        sb.nnet.pooling.Pooling2d(
            pool_type=cnn_pool_type,
            kernel_size=cnn_spatial_pool,
            stride=cnn_spatial_pool,
            pool_axis=[1, 2],
        ),
    )

    # Temporal separable convolution
    cnn_septemporal_kernels = (
        cnn_spatial_kernels * cnn_septemporal_depth_multiplier
    )
    self.conv_module.add_module(
        f'conv_{ConvModule.i}_2',
        sb.nnet.CNN.Conv2d(
            in_channels=cnn_spatial_kernels,
            out_channels=cnn_septemporal_kernels,
            kernel_size=cnn_septemporal_kernelsize,
            groups=cnn_spatial_kernels,
            padding="same",
            padding_mode="constant",
            bias=False,
            swap=True,
        ),
    )

    if cnn_septemporal_point_kernels is None:
        cnn_septemporal_point_kernels = cnn_septemporal_kernels

    self.conv_module.add_module(
        f'conv_{ConvModule.i}_3',
        sb.nnet.CNN.Conv2d(
            in_channels=cnn_septemporal_kernels,
            out_channels=cnn_septemporal_point_kernels,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
            swap=True,
        ),
    )
    self.conv_module.add_module(
        f'bnorm_{ConvModule.i}_3',
        sb.nnet.normalization.BatchNorm2d(
            input_size=cnn_septemporal_point_kernels,
            momentum=0.01,
            affine=True,
        ),
    )
    self.conv_module.add_module(f'act_{ConvModule.i}_3', activation)
    self.conv_module.add_module(
        f'pool_{ConvModule.i}_3',
        sb.nnet.pooling.Pooling2d(
            pool_type=cnn_pool_type,
            kernel_size=cnn_septemporal_pool,
            stride=cnn_septemporal_pool,
            pool_axis=[1, 2],
        ),
    )
    self.conv_module.add_module(f'dropout_{ConvModule.i}_3', torch.nn.Dropout(p=dropout))
    
    self.conv_module.add_module(
      "flatten", torch.nn.Flatten(),
    )

    ConvModule.i += 1
    
  def forward(self, x: Tensor) -> Tensor:
    """Returns the output of the model.

    Arguments
    ---------
    x : torch.Tensor (batch, time, EEG channel, channel)
        Input to convolve. 4d tensors are expected.
    """
    return self.conv_module(x)