# Semantic Segmentation

Lecturer: Silvia Cascianelli

## Introduction

* Learning Hierarchical Features for Scene Labeling - 2012 : sliding window approach, patch-based classification
* Fully Convolutional Networks for Semantic Segmentation - 2015 : ...

### Note on upsampling
The upsampling step can be done in different ways:
* Unpooling : reverse the max pooling operation, but it can be computationally expensive and may not capture fine details.
    * Bed of Nails unpooling : place the max-pooled value in the center of the upsampled region, and fill the rest with zeros. This can lead to sparse activations and may not capture fine details.
    * Nearest neighbor unpooling : simply repeat the max-pooled value to fill the upsampled region. This can lead to blocky artifacts and may not capture fine details.
    * Max unpooling : place the max-pooled value in the position of the maximum activation in the upsampled region. This can capture fine details, but it can be computationally expensive.
* Transposed convolution (also known as deconvolution) : learnable upsampling operation that can capture fine details, but it can be computationally expensive and may introduce checkerboard artifacts.