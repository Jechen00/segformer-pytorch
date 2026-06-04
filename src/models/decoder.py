#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Sequence, Optional

from src.models.modules import ConvBNAct
from src.ml_types import PythonNum


#####################################
# Classes
#####################################
class MLPDecoder(nn.Module):
    '''
    Multi-layer perceptron (MLP) decoder for SegFormer.
    This is implemented with convolutional layers instead of linear layers.
    
    Args:
        feature_dims (Sequence[int]): 
            Dimension of output features (channels or embeddings) in each encoder stage.
        fused_channels (int): 
            Number of channels used to project each encoder feature map to a unified channel dimension before fusing.
        num_classes (int): Number of segmentation classes.
        activation (optional, nn.Module): 
            Activation function for the convolutional layer used to fuse encoder feature maps.
            Default is `None`.
        channel_dropout_prob (PythonNum): 
            Probability of channel-wise dropout applied after fusing encoder feature maps.
            Entire feature channels are randomly zeroed during training.
            Default is `0.0`.
    '''
    def __init__(
        self, 
        feature_dims: Sequence[int], 
        fused_channels: int, 
        num_classes: int,
        activation: Optional[nn.Module] = None,
        channel_dropout_prob: PythonNum = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stage_convs = nn.ModuleList([
            nn.Conv2d(in_channels, fused_channels, kernel_size = 1)
            for in_channels in feature_dims
        ])
        
        self.fused_conv = ConvBNAct(len(feature_dims) * fused_channels, fused_channels, kernel_size = 1,
                                    include_bn = True, activation = activation)
        self.dropout = nn.Dropout2d(channel_dropout_prob)
        self.classifier = nn.Conv2d(fused_channels, num_classes, kernel_size = 1)
        
    def forward(self, encoder_outs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Args:
            encoder_outs (List[torch.Tensor]): 
                List of encoder output feature maps.
                The length of the list should match `feature_dims`.
                The i-th element is a tensor of shape `(batch_size, feature_dims[i], fmap_height, fmap_width)`.
            
        Returns:
            torch.Tensor: 
                Segmentation logit mask of shape `(batch_size, num_classes, fmap1_height, fmap1_width)`,
                where `(fmap1_height, fmap1_width)` is the spatial resolution of the first encoder stage feature map.
        '''
        fused_X = []
        for X, stage_conv in zip(encoder_outs, self.stage_convs):
            X = stage_conv(X)
            X = F.interpolate(
                X, 
                size = encoder_outs[0].shape[-2:], 
                mode = 'bilinear', 
                align_corners = False
            )
            fused_X.append(X)
        
        fused_X = torch.concatenate(fused_X, dim = 1)
        fused_X = self.dropout(self.fused_conv(fused_X))
        logit_mask = self.classifier(fused_X)
        return logit_mask