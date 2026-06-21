from .classification.mini_imagenet import MiniImageNet
from .classification.human_binary import HumanBinary
from .segmentation.supervisely_person_filtered import SuperviselyPersonFiltered

__all__ = [
    'MiniImageNet', 'HumanBinary',
    'SuperviselyPersonFiltered'
]