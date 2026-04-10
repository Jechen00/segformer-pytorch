#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from src.models.encoder import MixTransformer


#####################################
# Classes
#####################################
class MiTClassification(nn.Module):
    '''
    Image classification model with a Mix Transformer (MiT) backbone.
    It uses a simple classification head consisting of:
        1) Global average pooling
        2) Linear layer
    Only the final stage output (feature map) of the MiT backbone is used for classification.

    Args:
        mit (MixTransformer): The MiT backbone.
        mit_channels (int): Number of channels in the final stage output of mit.
        num_classes (int): Number of classes.
    '''
    def __init__(self, mit: MixTransformer, mit_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.mit = mit
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mit_channels, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
                              This should be compatible with the expected input for mit.

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        '''
        mit_features = self.mit(X)[-1] # Only uses final stage
        return self.classifier(mit_features)