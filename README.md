# Noise-learning

A Noise learning method for denoising hyperspectral Raman data.

The code is wrote in Python3.X with a PyTorch framework.

A novel convolutional neural network, AUnet, is used as the deep learing model to learn the instrumental noise of each hyperspectral Raman microscopy.

The AUnet use 1-D Unet as the backbone, and synergize the spatial and channel attention module to optimize the network performance.
