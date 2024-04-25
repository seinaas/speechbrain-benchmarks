"""EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
Shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals.
It was proposed for P300, error-related negativity, motor execution, motor imagery decoding.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb
from models.modules.PositionalEncoding import PositionalEncoding
from models.modules.ConvModule import ConvModule


class ConformerNet(torch.nn.Module):
    """ConformerNet.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
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
    transformer_d_model: int
        The dimensionality of the transformer embeddings.
    transformer_nhead
        The number of heads in the multiheadattention models.
    transformer_num_layers
        The number of sub-encoder-layers in the encoder.
    transformer_dim_feedforward
        The dimension of the feedforward network model.
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
        cnn_temporal_kernels=8,
        cnn_temporal_kernelsize=(33, 1),
        cnn_spatial_depth_multiplier=2,
        cnn_spatial_max_norm=1.0,
        cnn_spatial_pool=(4, 1),
        cnn_septemporal_depth_multiplier=1,
        cnn_septemporal_point_kernels=None,
        cnn_septemporal_kernelsize=(17, 1),
        cnn_septemporal_pool=(8, 1),
        cnn_pool_type="avg",
        dropout=0.5,
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
        
        # CONVOLUTIONAL MODULE
        self.conv_module = ConvModule(
            C=C,
            cnn_temporal_kernels=cnn_temporal_kernels,
            cnn_temporal_kernelsize=cnn_temporal_kernelsize,
            cnn_spatial_depth_multiplier=cnn_spatial_depth_multiplier,
            cnn_spatial_max_norm=cnn_spatial_max_norm,
            cnn_spatial_pool=cnn_spatial_pool,
            cnn_septemporal_depth_multiplier=cnn_septemporal_depth_multiplier,
            cnn_septemporal_point_kernels=cnn_septemporal_point_kernels,
            cnn_septemporal_kernelsize=cnn_septemporal_kernelsize,
            cnn_septemporal_pool=cnn_septemporal_pool,
            cnn_pool_type=cnn_pool_type,
            activation=activation,
            dropout=dropout,
        )

        # Shape of intermediate feature maps
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)
               
        # TRANSFORMER MODULE
        self.transformer_embedding = torch.nn.Linear(dense_input_size, transformer_d_model)
        self.pos_encoder = PositionalEncoding(transformer_d_model, dropout=dropout)
        self.transformer_module = torch.nn.Sequential()
        
        self.transformer_module.add_module("dropout_1", torch.nn.Dropout(p=dropout))
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        transformer_encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=transformer_num_layers)
        self.transformer_module.add_module(
            "enc_1",
            transformer_encoder,
        )
        
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=transformer_d_model,
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

        x_embed = self.transformer_embedding(x_conv) # Embedding -> (500, 32, 64)
        x_enc = self.pos_encoder(x_embed)
        
        x_transformer = self.transformer_module(x_enc)
        x_transformer = torch.mean(x_transformer, dim=0)
      
        x = self.dense_module(x_transformer)

        return x
