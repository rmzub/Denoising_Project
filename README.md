# Deep Denoising with Autoencoders (CIFAR-10)

This repository demonstrates how to use a **convolutional autoencoder** to denoise images from the **CIFAR-10** dataset. We compare the autoencoder's performance with a traditional **Gaussian filter** to highlight the benefits of learned approaches over simple smoothing techniques.

## Introduction
This project aims to reduce image noise using a **deep convolutional autoencoder**. Autoencoders learn to compress (encode) an image into a lower-dimensional space and then reconstruct (decode) a noise-free version of the image. The CIFAR-10 dataset (color images, 32×32) is used to train and evaluate the denoising performance.

**Key Objectives**:
- Train a CNN autoencoder to remove synthetic noise from CIFAR-10 images.
- Compare performance with a simple **Gaussian blur** as a baseline.
- Measure image quality using **PSNR** (Peak Signal-to-Noise Ratio).

![telegram-cloud-photo-size-2-5422657637725629289-y](https://github.com/user-attachments/assets/00ec9b57-7afd-4fe7-b849-b53b6b0c5c91)
