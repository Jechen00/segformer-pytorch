#####################################
# Imports & Dependencies
#####################################
import datasets
from datasets import ClassLabel
from typing import Optional, List, Callable, Sequence

from src.data_setup.datasets.classification.base import ClassificationDatasetBase
from src.data_setup.types import ClsSample


#####################################
# Hugging Face Dataset Wrapper
#####################################
class HFClassification(ClassificationDatasetBase):
    '''
    `ClassificationDatasetBase` wrapper around a Hugging Face classification dataset.
    This allows for optional per-sample image transforms.
    
    Tensor and Transform Notes:
        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.
          
        - If `transforms` are provided, it is recommended that the Hugging Face dataset (`hf_dataset`)
          does not have transforms attached to it 
          (either through `hf_dataset.set_transform()` or `hf_dataset.with_transform()`).
          This ensures that there is only a single source of transforms.

    Args:
        hf_dataset (datasets.Dataset): 
            Hugging Face dataset containing classification samples.
            Each sample must be a dictionary (ClsSample) containing the same keys.
            Required fields:
                - image (ImageInput): Image sample.
                - label (Union[int, torch.Tensor]): Class label for the image.
        transforms (optional, Callable): 
            Transform pipeline applied to each image.
            This must be compatible with the samples in `hf_dataset`.
            This must accept and return a sample dictionary (`ClsSample`).
        class_names (optional, List[str]): 
            List of class names, where index corresponds to label encoding in `hf_dataset`.
            If not provided, will try to fallback to `hf_dataset.features['label'].names` if available.
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
        hf_dataset: datasets.Dataset, 
        transforms: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None
    ):
        super().__init__(
            transforms = transforms, 
            to_tensor = to_tensor, 
            norm_mean = norm_mean, 
            norm_std = norm_std
        )
        self.hf_dataset = hf_dataset

        # Setup class names
        if class_names is not None:
            self.class_names = class_names
        else:
            feature_labels = hf_dataset.features.get('label')
            if isinstance(feature_labels, ClassLabel):
                self.class_names = feature_labels.names
            else:
                self.class_names = None

        if self.class_names is not None:
            self.class_to_idx, self.idx_to_class = {}, {}
            for i, label in enumerate(self.class_names):
                self.class_to_idx[label] = i
                self.idx_to_class[i] = label
        else:
            self.class_to_idx, self.idx_to_class = None, None

    def __len__(self) -> int:
        '''
        Returns the number of samples in the HF dataset.
        '''
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        '''
        Returns a representative string for the HF dataset.
        '''
        return self.hf_dataset.__repr__()
    
    def get_base_item(self, idx: int) -> ClsSample:
        '''
        Gets a single-sample dictionary directly from the HF dataset.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            ClsSample:
                Single-sample dictionary from `hf_dataset` containing:
                    - image (ImageInput): Image sample.
                    - label (ImageLabel): Class label for the image.
        '''
        return self.hf_dataset[idx].copy()