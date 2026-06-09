#####################################
# Imports & Dependencies
#####################################
import requests

import datasets
from pathlib import Path
from typing import Optional, Literal, List, Union, Callable, Sequence

from src.data_setup.datasets.classification import HFClassificationDataset
from src.data_setup.types import ClsSample


#####################################
# Dataset Class
#####################################
class MiniImageNetDataset(HFClassificationDataset):
    '''
    Wraps the Mini-ImageNet HF dataset from https://huggingface.co/datasets/timm/mini-imagenet.

    This applies the following to the HF dataset samples:
        - Optional user-provided transforms
        - Optional tensor and datatype conversion
        - Optional normalization

    Dataset Notes:
        - A sample from the Mini-ImageNet HF dataset is a dictionary containing:
            - image (Image.Image): PIL image with shape varying per sample.
            - label (int): Class index for the image sample.

    Tensor and Transform Notes:
        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.

    Args:
        root (Union[str, Path]): 
            Directory to download the dataset in. This is the `cache_dir` of the HF dataset.
        split (Literal['train', 'val', 'test']): 
            Whether to construct the training dataset (`train`), 
            validation dataset (`val`), or test dataset (`test`).
        transforms (optional, Callable): 
            Transform pipeline applied to each image.
            This must be compatible with the samples of the Mini-ImageNet dataset.
            This must accept and return a sample dictionary (`ClsSample`).
        to_tensor (bool): 
            Whether to apply tensor conversions to all samples **after** any provided `transforms`.
            Images are scaled to [0, 1] with dtype `torch.float32`.
            Labels are converted to dtype `torch.long`.
            Default is `True`.
        norm_mean (optional, Sequence[float]): 
            Sequence of means (one for each input channel)
            used to normalize images **after** tensor conversion.
            If provided, `norm_std` must also be provided.
        norm_std (optional, Sequence[float]): 
            Sequence of standard deviations (one for each input channel)
            used to normalize images **after** tensor conversion.
            If provided, `norm_mean` must also be provided.
    '''
    def __init__(
        self, 
        root: Union[str, Path],
        split: Literal['train', 'val', 'test'],
        transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None
    ):
        self.split = split
        self.root = Path(root)
        self.data_dir = self.root / 'timm___mini-imagenet'

        # Load Mini-ImageNet HF dataset
        self.hf_dataset = datasets.load_dataset(
            'timm/mini-imagenet',
            cache_dir = root,
            split = split if split != 'val' else 'validation'
        )

        class_names = self._get_class_names()

        # Initialize HFClassificationDataset
        super().__init__(
            hf_dataset = self.hf_dataset,
            class_names = class_names,
            transforms = transforms,
            to_tensor = to_tensor,
            norm_mean = norm_mean,
            norm_std = norm_std
        )

    def get_base_item(self, idx: int) -> ClsSample:
        '''
        Gets a single-sample dictionary from the Mini-ImageNet HF dataset.
        All PIL images are converted to RGB format.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            ClsSample:
                Single-sample dictionary containing:
                    - image (Image.Image): PIL image from the Mini-ImageNet HF dataset,
                                           converted to RGB format.
                    - label (int): Class label for from the Mini-ImageNet HF dataset.
        '''
        item = self.hf_dataset[idx].copy()
        item['image'] = item['image'].convert('RGB') # Convert all to RGB (3 channels)
        return item

    def _get_class_names(self) -> List[str]:
        '''
        Returns a list of class names for the Mini-ImageNet HF dataset
        The ordering of the list matches the index order of the dataset.
        '''
        # Fetch synset ID to class name mapping
        mapping_url = 'https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw'
        response = requests.get(mapping_url)
        response.raise_for_status()

        synset_to_class = {}
        for line in response.text.splitlines():
            labeling = line.strip().split()
            synset_to_class[labeling[0]] = labeling[2]

        self.synset_to_class = synset_to_class
        
        # Get class names from synset ids
        self.synset_ids = self.hf_dataset.features['label'].names
        return [synset_to_class[synset_id] for synset_id in self.synset_ids]