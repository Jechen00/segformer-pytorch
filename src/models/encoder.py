#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torchvision.ops import StochasticDepth

import numpy as np
from itertools import islice
from typing import Sequence, Union, List, Optional

from src.ml_types import SpatialSize 
from src.utils import make_tuple
from src.models.modules import ChannelwiseLayerNorm, EfficientSelfAttention, MixFFN


#####################################
# Classes
#####################################
class EncoderBlock(nn.Module):
    '''
    A single block in the SegFormer Encoder. 
    The structure is as follows:
        1) Layer norm -> efficient self-attention -> residual (with stochastic depth)
        2) Layer norm -> mix-FFN -> residual (with stochastic depth)
        
    Args:
        feature_dim (int): Dimension of input and output features (channels for feature maps or embeddings for tokens).
        num_heads (int): Number of attention heads for efficient self-attention.
        reduce_ratio (int): Ratio used to reduce the spatial resolution (sequence length) of keys/values
                            before computing efficient self-attention.
                            The sequence length is specifically reduced by a factor of roughly `reduce_ratio**2`.
                            If `reduce_ratio = 1`, no reduction is applied.      
        hid_dim (int): Number of hidden channels in intermediate layers of the mix-FFN.
        activation (optional, nn.Module): Activation function applied within the mix-FFN.
                                          If `None`, defaults to GELU.
        attn_dropout_prob (float): Dropout probability for the attention weights. Default is `0.0`.
        sdepth_prob (float): Stochastic depth probability to drop residuals in residual connections. Default is `0.0`.
    '''
    def __init__(
        self, 
        feature_dim: int,
        num_heads: int,
        reduce_ratio: int,
        hid_dim: int,
        activation: Optional[nn.Module] = None,
        attn_dropout_prob: float = 0.0,
        sdepth_prob: float = 0.0
    ):
        super().__init__()
        self.seq_attn = nn.Sequential(
            ChannelwiseLayerNorm(feature_dim),
            EfficientSelfAttention(num_heads, feature_dim, reduce_ratio, attn_dropout_prob)
        )
        self.seq_mix_ffn = nn.Sequential(
            ChannelwiseLayerNorm(feature_dim),
            MixFFN(feature_dim, hid_dim, activation)
        )
        self.sdepth = StochasticDepth(sdepth_prob, mode = 'row')
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): Input tensor of shape `(batch_size, feature_dim, height, width)`.
            
        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, feature_dim, height, width)`.
        '''
        X = X + self.sdepth(self.seq_attn(X))
        X = X + self.sdepth(self.seq_mix_ffn(X))
        return X
    

class EncoderStage(nn.Module):
    '''
    A single stage of the SegFormer Encoder.
    The structure is as follows:
        1) Patch embedding
            - Non-overlapping: `stride == patch_size`
            - Overlapping: `stride < patch_size`
        2) Sequence of encoder blocks (self-attention -> mix-FFN)
        
    Args:
        in_channels (int): Number of input channels.
        feature_dim (int): Dimension of output features (channels for feature maps or embeddings for tokens).
        patch_size (int or tuple(int, int)): Spatial size of each patch region used to compute a embedding.
                                             This is the kernel size of a convolutional layer.
                                             If `int`, assumed square.
        stride (int or tuple(int, int)): Step size between patch regions.
                                         If `int`, assumed square.
        num_blks (int): Number of encoder blocks.
        num_heads (int): Number of attention heads for efficient self-attention.
        reduce_ratio (int): Ratio used to reduce the spatial resolution (sequence length) of keys/values
                            before computing efficient self-attention.
                            Sequence length is reduced by roughly `reduce_ratio**2`.
                            If `reduce_ratio = 1`, no reduction is applied.
        hid_dim (int): Number of hidden channels in intermediate layers of the mix-FFN.
        activation (optional, nn.Module): Activation function applied within the mix-FFN.
                                          If `None`, defaults to GELU.
        attn_dropout_prob (float): Dropout probability for the attention weights. Default is `0.0`.
        sdepth_probs (float or Sequence[float]): Stochastic depth probability for each encoder block.
                                                 If `Sequence[float]`, the length must equal `num_blks`.
                                                 If `float`, the same value is used for all encoder blocks.
                                                 Default is `0.0`.
    '''
    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        patch_size: SpatialSize,
        stride: SpatialSize,
        num_blks: int,
        num_heads: int,
        reduce_ratio: int,
        hid_dim: int,
        activation: Optional[nn.Module] = None,
        attn_dropout_prob: float = 0.0,
        sdepth_probs: Union[float, Sequence[float]] = 0.0 
    ):
        super().__init__()
        if isinstance(sdepth_probs, float):
            sdepth_probs = [sdepth_probs] * num_blks
        elif len(sdepth_probs) != num_blks:
            raise ValueError('Length of sdepth_probs must equal num_blks.')
            
        patch_size = make_tuple(patch_size)
        self.patch_embed = nn.Conv2d(in_channels, feature_dim, 
                                     kernel_size = patch_size, stride = stride, 
                                     padding = (patch_size[0]//2, patch_size[1]//2))
        self.layer_norm1 = ChannelwiseLayerNorm(feature_dim)
        self.blocks = nn.ModuleList([
            EncoderBlock(feature_dim, num_heads, reduce_ratio, hid_dim, activation, 
                         attn_dropout_prob, sdepth_probs[i])
            for i in range(num_blks)
        ])
        self.layer_norm2 = ChannelwiseLayerNorm(feature_dim)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): Input tensor of shape `(batch_size, in_channels, height, width)`.
            
        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, feature_dim, fmap_height, fmap_width)`.
        '''
        X = self.layer_norm1(self.patch_embed(X))
        for block in self.blocks:
            X = block(X)
            
        return self.layer_norm2(X)
    

class MixTransformer(nn.Module):
    '''
    The Mix Transformer (MiT) encoder for SegFormer.
    
    Args:
        in_channels (int): Number of input channels.
        feature_dims (Sequence[int]): Dimension of output features (channels or embeddings) in each encoder stage.
        patch_sizes (Sequence[SpatialSize]): Patch size for the patch embedding in each encoder stage.
        strides (Sequence[SpatialSize]): Stride for the patch embedding in each encoder stage.
        num_blks (Sequence[int]): Number of encoder blocks for each encoder stage.
        num_heads (Sequence[int]): Number of attention heads for the efficient self-attention in each encoder stage.
        reduce_ratios (Sequence[int]): Reduction ratio for the efficient self-attention in each encoder stage.
        hid_dims (Sequence[int]): Hidden dimension of the mix-FFN in each encoder stage.
        activations (optional, Sequence[nn.Module]): Activation function for the mix-FFN in each encoder stage.
                                                     If `None`, defaults to GELU for each mix-FFN.
        attn_dropout_probs (optional, Sequence[float]): Dropout probability for the attention weights
                                                        in each encoder stage.
                                                        If `None`, defaults to `0.0` for all encoder stages.
        max_sdepth_prob (float): Maximum stochastic depth probability.
                                 The probability starts at `0.0` and linearly increases across all encoder blocks,
                                 reaching `max_sdepth_prob` at the final block of the final stage.
                                 Default is `0.0`.
                                 
    Note: All sequence inputs must have the same length.
    '''
    def __init__(
        self,
        in_channels: int,
        feature_dims: Sequence[int],
        patch_sizes: Sequence[SpatialSize],
        strides: Sequence[SpatialSize],
        num_blks: Sequence[int],
        num_heads: Sequence[int],
        reduce_ratios: Sequence[int],
        hid_dims: Sequence[int],
        activations: Optional[Sequence[nn.Module]] = None,
        attn_dropout_probs: Optional[Sequence[float]] = None,
        max_sdepth_prob: float = 0.0
    ):
        super().__init__()
        num_stages = len(feature_dims)
        if activations is None:
            activations = [nn.GELU() for _ in range(num_stages)]
        if attn_dropout_probs is None:
            attn_dropout_probs = [0.0] * num_stages
            
        stage_fields = [
            ('in_channels', [in_channels, *feature_dims[:-1]]),
            ('feature_dims', feature_dims),
            ('patch_sizes', patch_sizes),
            ('strides', strides),
            ('num_blks', num_blks),
            ('num_heads', num_heads),
            ('reduce_ratios', reduce_ratios),
            ('hid_dims', hid_dims),
            ('activations', activations),
            ('attn_dropout_probs', attn_dropout_probs)
        ]
        
        # Check each field has the same length
        for name, val in stage_fields:
            if len(val) != num_stages:
                raise ValueError(
                    f'{name} must have the same length as feature_dim (equal to the number of encoder stages). '
                    f'Got len({name}) = {len(val)}, but expected {num_stages}.'
                )
                
        # Get list of stochastic depth probs for each stage
        iter_sdepth_probs = iter(np.linspace(0, max_sdepth_prob, sum(num_blks)))
        
        # Create encoder stages
        _, stage_field_vals = zip(*stage_fields)
        self.stages = nn.ModuleList([])
        for i, stage_args in enumerate(zip(*stage_field_vals)):
            sdepth_probs = list(islice(iter_sdepth_probs, num_blks[i]))
            self.stages.append(EncoderStage(*stage_args, sdepth_probs))
        
    def forward(self, X: torch.Tensor) -> List[torch.Tensor]:
        '''
        Args:
            X (torch.Tensor): Input tensor of shape `(batch_size, in_channels, height, width)`.
            
        Returns:
            List[torch.Tensor]: List of output tensors from each encoder stage.
                                The i-th tensor has shape `(batch_size, feature_dim[i], fmap_height, fmap_width)`.
        '''
        outs = []
        for stage in self.stages:
            X = stage(X)
            outs.append(X)
        return outs