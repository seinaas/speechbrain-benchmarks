import torch
import torch.nn as nn
import math
from torch import Tensor

class PositionalEncoding(nn.Module):
  """Positional encoding module for transformer embeddings.

    Arguments
    ---------
    d_model: int
        The dimensionality of the transformer embeddings.
    dropout: float
        Dropout probability.
    max_len: int
        Maximum length of the encoding.

    Example
    -------
    #>>> inp_tensor = torch.rand([200, 1, 32])
    #>>> pos_enc = PositionalEncoding(d_model=32)
    #>>> output = pos_enc(inp_tensor)
    """
    
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = torch.nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)