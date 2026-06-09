from .classification.mini_imagenet import MiniImageNetDataset
from .classification.human_binary import HumanBinaryDataset
from .segmentation.supervisely_person import SuperviselyPersonDataset

__all__ = [
    'MiniImageNetDataset', 'HumanBinaryDataset',
    'SuperviselyPersonDataset'
]