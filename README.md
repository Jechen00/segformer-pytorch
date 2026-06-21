# An Unofficial PyTorch Reimplementation of SegFormer
This project uses the PyTorch framework to reimplement the SegFormer architecture from the paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203). Additionally, it provides a flexible training framework that supports both encoder pretraining and full SegFormer model training.

<p align = 'center'>
  <img src = 'images/demo_image_1.png' alt = 'SegFormer Demo 1' height = 320/>
  <img src = 'images/demo_image_2.png' alt = 'SegFormer Demo 2' height = 320/>
</p>

## Table of Contents
- [SegFormer Overview](#segformer-overview)
- [Reimplementation Details](#reimplementation-details)
- [Recommended Installation Instructions](#recommended-installation-instructions)
- [Example Training](#example-training)
- [References](#references)

## SegFormer Overview
SegFormer is a semantic segmentation architecture that was introduced by [Xie et al. (2021)](#references), consisting of a hierarchical transformer-based encoder and a simple multi-layer perceptron (MLP) decoder. In terms of both accuracy and efficiency, this novel encoder-decoder design enabled SegFormer to outperform many contemporary state-of-the-art architectures, including CNN-based methods like DeepLabV3+ and other transformer-based methods like PVT, Swin Transformer, and SETR.

### Encoder
The SegFormer encoder is known as the Mix Transformer (MiT) and its role is to produce various information-rich feature maps for the decoder. This is achieved through a hierarchy of stages (usually 4) that is conceptually similar to a feature pyramid network (FPN). Each encoder stage recieves the output of the previous stage (with the first stage directly processing the input image) and produces a feature map at a lower resolution. This results in feature maps at multiple scales with varying levels of spatial and semantic information, allowing the decoder to work with a more comprehensive representation of the input image. 

Each encoder stage first uses an overlapping patch embedding layer that converts their input into a sequence of patch embeddings (or tokens). The overlapping aspect of this layer produces neighboring patch embeddings with shared information at their boundaries, leading to smoother feature changes between them. The sequence of patch embeddings is then passed through a series of consecutive blocks, each consisting of an efficient self-attention layer followed by a feed-forward network (FFN). The efficient self-attention layer is the "Transformer" component of SegFormer and captures the contextual relationships across patch embeddings while using a sequence reduction process to reduce computational complexity. The subsequent FFN mixes a depth-wise convolutional layer into a MLP (called the Mix-FFN), which introduces additional positional information into the sequence while still transforming its features. Lastly, the sequence is reshaped back to a feature map, which is then output by the encoder stage.

### Decoder
The SegFormer decoder is a lightweight MLP whose role is to process the encoder feature maps into a logit map used for the final predicted segmentation mask. The decoder first applies a linear projection and upsamples each feature map, bringing them to a common spatial resolution (usually a fourth of the original image resolution) and channel dimension. The feature maps are then concatenated along their channel dimension to form a single representation that combines their spatial and semantic information. This unified feature map is transformed through the rest of the layers in the MLP to produce a logit map. Note that a final upsampling step is necessary to bring the logit map back to the original image resolution.

## Reimplementation Details
To be completely clear, this project is an **unofficial PyTorch reimplementation** of SegFormer, created mainly for exploration and to gain a deeper understanding of the architecture. The authors of SegFormer, [Xie et al. (2021)](#references), have released an official PyTorch implementation and can be found here: [SegFormer](https://github.com/NVlabs/SegFormer). In the making of this project, the official implementation was frequently used as a reference for the parts of the architecture that were not explicitly mentioned in the paper (e.g. layer normalizations, stochastic depth, dropout, and residual connections).

That being said, many aspects of this reimplementation differ from the official implementation. For instance, several changes were made with the goal of reducing the need to repeatedly reshape tensors between feature map representations (used for convolutional layers) and flattened sequence representations (used for linear layers and attention). This includes:
  - Replacing nearly all linear layers in the original architecture with equivalent 1x1 convolutional layers
  - Introducing a channel-wise layer normalization layer to apply layer normalization directly on feature maps
  - Using a convolutional layer for the sequence reduction process in efficient self-attention

In addition to the architecture reimplementation, this project also includes a dedicated training pipeline that can be used to pretrain the MiT encoder as well as train the full SegFormer model. The training setup does not follow the pipeline outlined in [Xie et al. (2021)](#references) or in the official implementation. Instead, it uses a different set of data augmentations (geometric and photometric), loss functions (CE-Dice loss for multiclass and Focal-Dice loss for binary), and learning rate schedulers (e.g. cosine annealing).

## Recommended Installation Instructions
### 1) Create a New Python Environment
This environment should use **Python >= 3.11**.

### 2) Clone the `segformer-pytorch` Repository
```
git clone https://github.com/Jechen00/segformer-pytorch.git
```

### 3) Install Required Packages
Navigate to the repository root directory (usually called `segformer-pytorch`) and run:
```
pip install -r requirements.txt
```

Alternatively, you may install the packages manually:
```
pip install matplotlib==3.11.0
pip install numpy==2.4.6
pip install opencv_python==4.13.0.92
pip install pandas==3.0.3
pip install Pillow==12.2.0
pip install PyYAML==6.0.3
pip install Requests==2.34.2
pip install seaborn==0.13.2
pip install torch==2.11.0
pip install torchvision==0.26.0
pip install transformers==5.8.1
```

### 4) (Optional) Download Pretrained Backbone Weights
When training a SegFormer model, it is typically recommended to first pretrain the MiT encoder/backbone on an image classification task with a large, general dataset. This provides the encoder with a strong starting point that can already extract meaningful features/patterns from images. As a result, training the SegFormer model often becomes faster and may lead to better performance.

The dataset that [Xie et al. (2021)](#references) used for pretraining is ImageNet-1k. However, since this project is more resource-limited, the available pretrained weights have only been pretrained on a small subset of ImageNet-1k known as [Mini-ImageNet](https://huggingface.co/datasets/timm/mini-imagenet), which contains 100 out of the 1000 classes and has fewer examples per class. Additionally, the pretrained weights are currently only provided for the MiT-B0 and MiT-B1 encoders.

The pretrained weights can be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/1CWVqNr0BeKhJChCLgeLsEyP_XQ9VLKJL?usp=drive_link). The pretrained weights for MiT-B0 and MiT-B1 are located in `mini_imagenet/mit_b0/best_backbone.pth` and `mini_imagenet/mit_b1/best_backbone.pth`, respectively. Feel free to download them and store them anywhere locally.

See [Example Training](#example-training) for links to the Jupyter notebooks used to pretrain the MiT encoders.

## Example Training
### Training Scripts
Example training scripts for encoder pretraining and full SegFormer training can be found in the [`training/backbone_pretrain`](./training/backbone_pretrain) and [`training/segformer`](./training/segformer) folders, respectively. The example encoder pretraining script pretrains a MiT-B1 encoder on the [Mini-ImageNet](https://huggingface.co/datasets/timm/mini-imagenet) dataset. The example SegFromer training script trains a SegFormer-B1 model on a [subset of the Supervisely Person dataset](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset), which is a binary segmentation dataset that contains human and background classes.

To run these scripts:
  1. If necessary, edit the `config.yaml` file to your specifications. Note that if you are using the segformer training script, you will need to download the MiT-B1 pretrained weights (see [Download Pretrained Backbone Weights](#4-optional-download-pretrained-backbone-weights)) and update the `mit_weights` field of `config.yaml` to point to the location of the downloaded weights.
     
  2. From the repository root directory, start training by running:
     ```
     python training/SCRIPT_DIR/run_training.py config.yaml
     ```
     where `SCRIPT_DIR` is either `backbone_pretrain` or `segformer`.
     
 ### Jupyter Notebooks
  - MiT-B0 Pretraining on Mini-ImageNet: [mit_b0_pretraining.ipynb](./notebooks/mit_b0_pretraining.ipynb)
  - MiT-B1 Pretraining on Mini-ImageNet: [mit_b1_pretraining.ipynb](./notebooks/mit_b1_pretraining.ipynb)
  - SegFormer-B1 Training on Subset of Supervisely Person: [segformer_b1_training.ipynb](./notebooks/segformer_b1_training.ipynb)

## References
Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." _arXiv_, 31 May 2021, [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203).

NVlabs, _SegFormer_, commit 65fa8cf, GitHub, 13 Jun. 2023, [https://github.com/NVlabs/SegFormer](https://github.com/NVlabs/SegFormer).