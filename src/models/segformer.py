#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from typing import Sequence, Optional

from src.utils.common_types import SpatialSize
from src.models.encoder import MixTransformer
from src.models.decoder import MLPDecoder


#####################################
# Model Classes
#####################################
class EncoderDecoder(nn.Module):
    '''
    Generic encoder-decoder architecture.

    Args:
        encoder (nn.Module): Encoder that processes an input tensor into one or more representations.
        decoder (nn.Module): Decoder that processes the encoder output and produces a final output tensor.
                             The expected input of this module must match the encoder output.
    '''
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): Input tensor expected by the encoder.

        Returns:
            torch.Tensor: Output tensor produced by the decoder.
        '''
        return self.decoder(self.encoder(X))
    

class SegFormerDefault(EncoderDecoder):
    '''
    Default SegFormer model with a Mix Transformer (Mit) encoder 
    and a multi-layer perceptron (MLP) decoder.
    The architecture follows the SegFormer paper: https://arxiv.org/abs/2105.15203

    Args:
        in_channels (int): Number of input channels

        feature_dims (Sequence[int]): Dimension of output features (channels or embeddings) in each encoder stage.
        patch_sizes (Sequence[SpatialSize]): Patch size for the patch embedding in each encoder stage.
        strides (Sequence[SpatialSize]): Stride for the patch embedding in each encoder stage.
        num_blks (Sequence[int]): Number of encoder blocks for each encoder stage.
        num_heads (Sequence[int]): Number of attention heads for the efficient self-attention in each encoder stage.
        reduce_ratios (Sequence[int]): Reduction ratio for the efficient self-attention in each encoder stage.
        hid_dims (Sequence[int]): Hidden dimension of the mix-FFN in each encoder stage.
        enc_activations (optional, Sequence[nn.Module]): Activation function for the mix-FFN in each encoder stage.
                                                         If None, defaults to GELU for each mix-FFN.
        attn_dropout_probs (optional, Sequence[float]): Dropout probability for the attention weights
                                                        in each encoder stage.
                                                        If None, defaults to 0.0 for all encoder stages.
        max_sdepth_prob (float): Maximum stochastic depth probability.
                                 The probability starts at 0.0 and linearly increases across all encoder blocks,
                                 reaching max_sdepth_prob at the final block of the final stage.
                                 Default is 0.0.
        
        fused_channels (int): Number of channels used to project each encoder feature map
                              to a unified channel dimension before fusing.
        num_classes (int): Number of segmentation classes.
        dec_activation (optional, nn.Module): Activation function for the convolutional layer
                                              used to fuse encoder feature maps in the decoder.
                                              Default is None.
        channel_dropout_prob (float): Probability of channelwise dropout applied 
                                      after fusing encoder feature maps in the decoder.
                                      Entire feature channels are randomly zeroed during training.
                                      Default is 0.0.
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

        fused_channels: int, 
        num_classes: int,

        enc_activations: Optional[Sequence[nn.Module]] = None,
        attn_dropout_probs: Optional[Sequence[float]] = None,
        max_sdepth_prob: float = 0.0,

        dec_activation: Optional[nn.Module] = None,
        channel_dropout_prob: float = 0.0
    ):
        encoder = MixTransformer(in_channels, feature_dims, patch_sizes, strides,
                                 num_blks, num_heads, reduce_ratios, hid_dims,
                                 enc_activations, attn_dropout_probs, max_sdepth_prob)
        
        decoder = MLPDecoder(feature_dims, fused_channels, num_classes,
                             dec_activation, channel_dropout_prob)
        
        super().__init__(encoder, decoder)
