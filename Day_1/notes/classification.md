# Classification problems

Lecturer: Sara Sarto

## Intro

Datasets:
* MNIST
* CIFAR-10 / CIFAR-100
* ImageNet

## History

* alexnet - 2012
* vgg - 2014: very deep convolutional network
* googlenet - 2014 : inception module
* resnet - 2015 : residual connections
* Network in Network (NiN) - 2014
* Identity Mappings in Deep Residual Networks - 2016
* Wide Residual Networks - 2016 : wider layers instead of deeper (more parallelism)
* ResNeXt - 2017 : split-transform-merge strategy, combines the ideas of ResNet and GoogLeNet
* Deep Networks with Stochastic Depth - 2016 : randomly drops some layers during training to improve generalization
* Good Practices for Deep Feature Fusion - 2016
* SENet - 2017 : Squeeze-and-Excitation Networks, introduces a channel-wise attention mechanism
* FractalNet - 2017 : uses a fractal architecture to improve feature representation, with both shallow and deep paths to output, as an alternative to residual connections.
* Dense Connected Convolutional Networks (DenseNet) - 2017 : each layer is connected to every other layer in a feed-forward fashion, improving gradient flow and feature reuse.
* SqueezeNet - 2017 : a smaller architecture that achieves AlexNet-level accuracy with 50x fewer parameters, using fire modules to reduce the number of parameters.

Self-attentive architectures:
* ViT - 2020 : Vision Transformer, applies the transformer architecture to image classification tasks
* Scalable visual transformers with hierarchical pooling - 2021 : introduces a hierarchical pooling mechanism to reduce the computational cost of ViT while maintaining performance.

Multimodal architectures:
* CLIP - 2021 : Contrastive Language-Image Pre-training, learns visual representations from natural language supervision, enabling zero-shot transfer to downstream tasks.