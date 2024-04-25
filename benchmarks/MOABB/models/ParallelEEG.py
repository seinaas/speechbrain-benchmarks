"""ParallelEEG
Parallel Convolutional and Transformer model for classifying EEG data.

Authors
 * Seina Assadian, 2024
"""
import torch
import speechbrain as sb
from models.modules.PositionalEncoding import PositionalEncoding
from models.modules.ConvModule import ConvModule

class ParallelEEG(torch.nn.Module):
    """ParallelEEG.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_broad_temporal_kernels: int
        Number of kernels in the 2d temporal convolution (broad CNN).
    cnn_broad_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution (broad CNN).
    cnn_broad_spatial_depth_multiplier: int
        Depth multiplier of the 2d spatial depthwise convolution (broad CNN).
    cnn_broad_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution (broad CNN).
    cnn_broad_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution (broad CNN).
    cnn_broad_septemporal_depth_multiplier: int
        Depth multiplier of the 2d temporal separable convolution (broad CNN).
    cnn_broad_septemporal_kernelsize: tuple
        Kernel size of the 2d temporal separable convolution (broad CNN).
    cnn_broad_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution (broad CNN).
    cnn_fine_temporal_kernels: int
        Number of kernels in the 2d temporal convolution (fine CNN).
    cnn_fine_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution (fine CNN).
    cnn_fine_spatial_depth_multiplier: int
        Depth multiplier of the 2d spatial depthwise convolution (fine CNN).
    cnn_fine_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution (fine CNN).
    cnn_fine_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution (fine CNN).
    cnn_fine_septemporal_depth_multiplier: int
        Depth multiplier of the 2d temporal separable convolution (fine CNN).
    cnn_fine_septemporal_kernelsize: tuple
        Kernel size of the 2d temporal separable convolution (fine CNN).
    cnn_fine_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution (fine CNN).
    cnn_pool_type: string
        Pooling type.
    transformer_pool_type: string
        Pooling type for the transformer module.
    transformer_pool: tuple
        Pool size and stride before the transformer module.
    transformer_d_model: int
        The dimensionality of the transformer embeddings.
    transformer_nhead
        The number of heads in the multiheadattention models.
    transformer_num_layers
        The number of sub-encoder-layers in the encoder.
    transformer_dim_feedforward
        The dimension of the feedforward network model.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        cnn_broad_temporal_kernels=8,
        cnn_broad_temporal_kernelsize=(33, 1),
        cnn_broad_spatial_depth_multiplier=2,
        cnn_broad_spatial_max_norm=1.0,
        cnn_broad_spatial_pool=(4, 1),
        cnn_broad_septemporal_depth_multiplier=1,
        cnn_broad_septemporal_point_kernels=None,
        cnn_broad_septemporal_kernelsize=(17, 1),
        cnn_broad_septemporal_pool=(8, 1),
        cnn_fine_temporal_kernels=8,
        cnn_fine_temporal_kernelsize=(17, 1),
        cnn_fine_spatial_depth_multiplier=2,
        cnn_fine_spatial_max_norm=1.0,
        cnn_fine_spatial_pool=(4, 1),
        cnn_fine_septemporal_depth_multiplier=1,
        cnn_fine_septemporal_point_kernels=None,
        cnn_fine_septemporal_kernelsize=(7, 1),
        cnn_fine_septemporal_pool=(4, 1),
        cnn_pool_type="avg",
        dropout=0.5,
        transformer_pool_type='max',
        transformer_pool=(4, 1),
        transformer_d_model=64,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dim_feedforward=256,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="elu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        # T = input_shape[1]
        C = input_shape[2]
        
        # TRANSFORMER MODULE
        self.transformer_maxpool = sb.nnet.pooling.Pooling2d(
                pool_type=transformer_pool_type,
                kernel_size=transformer_pool,
                stride=transformer_pool,
                pool_axis=[1, 2],
            )
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        
        self.transformer_module = torch.nn.Sequential()
        self.transformer_module.add_module(
            'embed_0',
            torch.nn.Linear(C, transformer_d_model),
        )
        
        self.transformer_module.add_module(
            'pos_enc',
            PositionalEncoding(d_model=transformer_d_model, dropout=dropout),
        )
        
        self.transformer_module.add_module(
          'transformer_encoder',
          torch.nn.TransformerEncoder(transformer_layer, num_layers=transformer_num_layers),
        )

        # CONVOLUTIONAL MODULE (BROAD FEATURES)
        self.conv_module = ConvModule(
            C=C,
            cnn_temporal_kernels=cnn_broad_temporal_kernels,
            cnn_temporal_kernelsize=cnn_broad_temporal_kernelsize,
            cnn_spatial_depth_multiplier=cnn_broad_spatial_depth_multiplier,
            cnn_spatial_max_norm=cnn_broad_spatial_max_norm,
            cnn_spatial_pool=cnn_broad_spatial_pool,
            cnn_septemporal_depth_multiplier=cnn_broad_septemporal_depth_multiplier,
            cnn_septemporal_point_kernels=cnn_broad_septemporal_point_kernels,
            cnn_septemporal_kernelsize=cnn_broad_septemporal_kernelsize,
            cnn_septemporal_pool=cnn_broad_septemporal_pool,
            cnn_pool_type=cnn_pool_type,
            activation=activation,
            dropout=dropout,
        )
       
        # CONVOLUTIONAL MODULE (FINE FEATURES)
        self.conv_module_2 = ConvModule(
            C=C,
            cnn_temporal_kernels=cnn_fine_temporal_kernels,
            cnn_temporal_kernelsize=cnn_fine_temporal_kernelsize,
            cnn_spatial_depth_multiplier=cnn_fine_spatial_depth_multiplier,
            cnn_spatial_max_norm=cnn_fine_spatial_max_norm,
            cnn_spatial_pool=cnn_fine_spatial_pool,
            cnn_septemporal_depth_multiplier=cnn_fine_septemporal_depth_multiplier,
            cnn_septemporal_point_kernels=cnn_fine_septemporal_point_kernels,
            cnn_septemporal_kernelsize=cnn_fine_septemporal_kernelsize,
            cnn_septemporal_pool=cnn_fine_septemporal_pool,
            cnn_pool_type=cnn_pool_type,
            activation=activation,
            dropout=dropout,
        )

        # Shape of intermediate feature maps
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        
        out_2 = self.conv_module_2(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out) + self._num_flat_features(out_2) + transformer_d_model
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        
        # x shape -> (batch, time, EEG channel, channel) -> (32, 500, 17, 1)
        x_conv = self.conv_module(x)
        x_conv_2 = self.conv_module_2(x)

        x_maxpool = self.transformer_maxpool(x) # Maxpool across time -> (32, 125, 17, 1)
        x_maxpool = torch.squeeze(x_maxpool, 3).permute(1, 0, 2) # Drop channel dimension and permute -> (125, 32, 17)
        x_transformer = self.transformer_module(x_maxpool)
        x_transformer = torch.mean(x_transformer, dim=0)
        
        x_final = torch.cat([x_conv, x_conv_2, x_transformer], 1)
        
        x = self.dense_module(x_final)

        return x
