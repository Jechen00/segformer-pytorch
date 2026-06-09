#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional

from src.ml_types import PythonNum

ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'mish': nn.Mish,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid
}


#####################################
# Classes
#####################################
class ConvBNAct(nn.Module):
    '''
    Creates a block: convolutional layer -> optional batch normalization -> optional activation.

    Args:
        in_channels (int): 
            Number of input channels for the conv layer.
        out_channels (int):     
            Number of output channels for the conv layer.
        kernel_size (int): 
            Kernel size for the conv layer.
        stride (int): 
            Stride for the conv layer. Default is `1`.
        padding (int): 
            Padding for the conv layer. Default is `0`.
        include_bn (bool): 
            Whether to include batch norm after the conv layer. Default is `False`.
        activation (optional, nn.Module): 
            Activation function applied after the conv layer (and batch norm if included).
            Default is `None`.
    '''
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 include_bn: bool = False, 
                 activation: Optional[nn.Module] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_bn = include_bn

        include_bias = not include_bn
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = include_bias)]
        
        if include_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if activation:
            layers.append(activation)
            
        self.conv_bn_act = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv_bn_act(X)
    

class ChannelwiseLayerNorm(nn.Module):
    '''
    Applies channel-wise layer normalization to a tensor of shape `(batch_size, num_channels, height, width)`.
    This normalizes across the channel dimension, independently at each spatial location `(height, width)`.
    This is done without permutating to channel-last format.

    Note: The channel variances are computed using the unbiased estimator, following PyTorch's implementation.

    Args:
        num_channels (int): 
            Number of input channels.
        eps (float): 
            Small value added to denominator of the layer norm to prevent numerical errors. 
            Default is `1e-6`.
    '''
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.biases = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): 
                Input tensor of shape `(batch_size, num_channels, height, width)`.
            
        Returns:
            torch.Tensor: 
                Output after applying layer norm on the `num_channels` dimension of `X`.
                Shape is `(batch_size, num_channels, height, width)`.
        '''
        mean = X.mean(dim = 1, keepdim = True)
        var = X.var(dim = 1, unbiased = False, keepdim = True)
        
        X = (X - mean) / (var + self.eps).sqrt()
        return X * self.weights + self.biases
    

class EfficientSelfAttention(nn.Module):
    '''
    Efficient self-attention implemented with convolutional layers instead of linear layers.
    The spatial resolution of queries is preserved, 
    while that of keys/values is reduced to improve computational efficiency.
    
    Introduced in the Pyramid Vision Transformer (PVT) paper: https://arxiv.org/abs/2102.12122
    
    Note: This implementation assumes the projected dimensions 
          for the queries, keys, and values are all the same `(d_q = d_k = d_v) `
          and equal to `feature_dim // num_heads`. 
          Consequently, it is required that `feature_dim` is divisible by `num_heads`.

    Args:
        num_heads (int): 
            Number of attention heads.
        feature_dim (int): 
            Number of features (channels for feature maps or embeddings for tokens).
        reduce_ratio (int): 
            Ratio used to reduce the spatial resolution (sequence length) of the keys/values
            before computing attention. 
            Sequence length is reduced by roughly `reduce_ratio**2`.
            If `reduce_ratio = 1`, no reduction is applied. Default is `1`.
        dropout_prob (PythonNum): 
            Dropout probability for the attention weights. Default is `0.0`.
    '''
    def __init__(self, num_heads: int, feature_dim: int, reduce_ratio: int = 1, dropout_prob: PythonNum = 0.0):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError('feature_dim must be divisible by num_heads.')
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.reduce_ratio = reduce_ratio
        self.dropout_prob = dropout_prob
        
        self.head_dim = feature_dim // num_heads # Assumed same for queries, keys, and values
        
        self.q_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size = 1, bias = False)
        if reduce_ratio > 1:
            self.reduce_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size = reduce_ratio, stride = reduce_ratio)
            self.layernorm = ChannelwiseLayerNorm(feature_dim)

        self.kv_conv = nn.Conv2d(feature_dim, 2 * feature_dim, kernel_size = 1, bias = False)
        self.out_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size = 1)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): 
                Input tensor of shape `(batch_size, feature_dim, height, width)`.
            
        Returns:
            torch.Tensor: 
                Output tensor from applying efficient self-attention to `X` 
                and reshaping back to a feature map.
                Shape is `(batch_size, feature_dim, height, width)`.
        '''
        batch_size, _, height, width = X.shape
        
        # Shape: (batch_size, num_heads, num_queries, head_dim)
        q = self.q_conv(X).reshape(batch_size, self.num_heads, self.head_dim, -1).transpose(2, 3)

        if self.reduce_ratio > 1:
            # Shape: (batch_size, 2 * feature_dim, height_reduced, width_reduced)
            kv = self.layernorm(self.reduce_conv(X))
            kv = self.kv_conv(kv)
        else:
            # Shape: (batch_size, 2 * feature_dim, height, width)
            kv = self.kv_conv(X)
            
        kv = kv.reshape(batch_size, 2, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        k, v = kv # Shape: (batch_size, num_heads, num_keys, head_dim)

        # Shape: (batch_size, num_heads, num_queries, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_prob)
    
        # out = out.transpose(1, 2).flatten(2) # Shape: (batch_size, num_queries, feature_dim)
        # out = self.out_linear(out) # Shape: (batch_size, num_queries, feature_dim)

        # Reshape back to a feature map of shape (batch_size, feature_dim, height, width)
        out = out.transpose(2, 3).reshape(batch_size, self.feature_dim, height, width)
        out = self.out_conv(out)
        return out


class MixFFN(nn.Module):
    '''
    Mix Feed-Forward (Mix-FFN) used in the SegFormer encoder.
    This implementation replaces the linear layers with equivalent 1x1 convolutional layers.
    
    Args:
        feature_dim (int): 
            Number of features (channels for feature maps or embeddings for tokens).
        hid_dim (int): 
            Number of hidden channels in intermediate layers.
            In the original SegFormer paper, 
            this is equal to `feature_dim` multiplied by some expansion ratio.
        activation (optional, nn.Module): 
            Activation function applied after 3x3 convolutional layer.
            If `None`, defaults to GELU.
        dropout_prob (PythonNum): 
            Dropout probability applied after the activation 
            and after the last linear layer (1x1 convolutional layer).
            This follows from the original implementation.
    '''
    def __init__(
        self, 
        feature_dim: int, 
        hid_dim: int, 
        activation: Optional[nn.Module] = None,
        dropout_prob: PythonNum = 0.0
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU()

        self.feature_dim = feature_dim

        self.mix_ffn = nn.Sequential(
            nn.Conv2d(feature_dim, hid_dim, kernel_size = 1), # First linear layer
            nn.Conv2d(hid_dim, hid_dim, kernel_size = 3, padding = 'same', groups = hid_dim), # Depth-wise convolutional layer
            activation,
            nn.Dropout(p = dropout_prob),
            nn.Conv2d(hid_dim, feature_dim, kernel_size = 1), # Second linear layer
            nn.Dropout(p = dropout_prob)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): 
                Input tensor of shape `(batch_size, feature_dim, height, width)`.
            
        Returns:
            torch.Tensor: 
                Output tensor of shape `(batch_size, feature_dim, height, width)`.
        '''
        return self.mix_ffn(X)