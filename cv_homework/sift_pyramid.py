#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

from cv2 import resize, GaussianBlur, subtract, INTER_LINEAR, INTER_NEAREST
from numpy import array, log, sqrt, zeros, round

################
# 图像金字塔构建 #
################

def generate_base_image(image, sigma, assumed_blur):
    """
    通过将输入图像升采样一倍并且高斯模糊得到高斯金字塔的基图像。
    如果希望输出的基图像具有尺度sigma，那么我们就需要用到高斯函数的性质：
    两个尺度为sigma的高斯滤波先后执行等价于一个尺度为sqrt2*sigma的高斯滤波，
    即sigma^2=sigma_{1}^2+sigma_{2}^2。由于原图像按照经验值假设已经具有尺度0.5，
    那么如果希望输出的基图像具有尺度sigma，就需要对已升采样的图像进行尺度为sigma_diff的高斯滤波。
    """
    # 将原图像升采样
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    # 计算sigma_diff=sqrt(sigma^2-(2*assumed_blur)^2)，这里之所以要使用2*assumed_blur
    # 是因为原图像被升采样了，相当于尺度变成了2*assumed_blur
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    # 对升采样原图进行尺度为sigma_diff的高斯滤波，得到尺度为sigma的目标图像
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


def compute_number_of_octaves(image_shape):
    """
    计算高斯金字塔的octave组数。
    若令min(image_shape)/2^x=1，可解得x=log(min(image_shape))/log(2)
    意思是如果我们让图像一直降采样，那么何时图像会缩成最小以至于无法继续降采样？显然
    是尺寸为1时，即上面那个方程，可将x解出。但是，由于后续需要在尺度空间上进行极值点
    搜索，搜索邻域是3*3*3，即空间上3*3，尺度空间上3个尺度，因此至少保证图像降采样到最后
    尺寸不得小于3。因此我们令log(min(image_shape)) / log(2) - 1，多减个1，使得
    不会降采样到最后尺寸为1。最后应该向下取整。
    """
    return int(round(log(min(image_shape)) / log(2) - 1))


def generate_gaussian_kernels(sigma, num_intervals):
    """
    生成一系列的高斯核，为了能获得不同尺度的图片。
    高斯金字塔有numOctaves组，每一组里面有numIntervals+3层，在一组
    里面所有的图像都有一样的长和宽，但同一组里面图像的尺度是递增的，使用的
    核就是这个函数生成的。我们希望同一组各个图像的尺度依次为sigma, k*sigma, k^2*sigma,...
    而这里我们生成的gaussian_kernels实际上为array([sigma, sqrt(k^2-1)*sigma, k*sqrt(k^2-1)*sigma,...]),
    其中，k=2^(1/num_intervals)，而len(gaussian_kernels)=num_intervals + 3
    如果我们print(gaussian_kernels)就可以得到array([1.6, 1.22627, 1.54501, 1.94659, 2.45255, 3.09002])，
    可以看到第一个数就是sigma，也就是基图像尺度为1.6，而第二个图像尺度应该为
    sqrt(1.6^2+1.22627^2)=2.01587，而不是1.22627；然后第三个图像尺度为
    sqrt(2.01587^2+1.54501^2)=2.53984，第四个图像尺度为
    sqrt(2.53984^2+1.94659^2)=3.2，这是下一组的第一个图像的尺度...
    """
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)

    gaussian_kernels = zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(
            sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


def generate_gaussian_images(image, num_octaves, gaussian_kernels):
    """
    生成尺度空间上的高斯金字塔。
    """
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(
                image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base,
                       (int(octave_base.shape[1] / 2),
                        int(octave_base.shape[0] / 2)),
                       interpolation=INTER_NEAREST)
    return array(gaussian_images)


def generate_DoG_images(gaussian_images):
    """
    生成尺度空间上的高斯差分金字塔。
    """
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(
                gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            # ordinary subtraction will not work because the images are
            # unsigned integers
            dog_images_in_octave.append(subtract(second_image, first_image))
        dog_images.append(dog_images_in_octave)
    return array(dog_images)