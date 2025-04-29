# MFAN: Multi-level Feature Attention Network for Medical Image Segmentation

This repository provides the official implementation of the paper:

**Multi-level Feature Attention Network for Medical Image Segmentation**  
Published in *Expert Systems with Applications*, 2025.  
DOI: [10.1016/j.eswa.2024.125785](https://doi.org/10.1016/j.eswa.2024.125785)

**Abstract**: Network architectures deriving from the Unet framework and its convolutional neural network variants have garnered significant attention for their impressive feats in computer vision. However, the shallow-level details and deep-level semantic information are underutilized in these methods, leading to the model’s inability to adequately localize target regions. In this paper, we put forward a Multi-level Feature Attention Network, a novel method that cross-connects encoder and decoder features and focuses on multi-scale semantic features. Firstly, we extend UperNet using a hierarchical Swin Transformer with shifted windows, giving the network global modeling capabilities. Secondly, we introduce a Cross-connection Multi-level Attention module that connects encoder and decoder to refine the decoder’s output features and supplement detailed information. Finally, we employ a Pyramid Collaborative Attention (PCA) module to mine the encoder’s deepest semantic features across multiple scales. Our method establishes state-of-the-art performance on the ACDC, ISIC2017 and BUSI datasets, showcasing its exceptional capability in segmenting medical images.

## Citation

If you find this work helpful, please cite:

```bibtex
@article{zhang2025multi,
  title={Multi-level Feature Attention Network for medical image segmentation},
  author={Zhang, Yaning and Yin, Jianjian and Gu, Yanhui and Chen, Yi},
  journal={Expert Systems with Applications},
  volume={263},
  pages={125785},
  year={2025},
  publisher={Elsevier}
}
