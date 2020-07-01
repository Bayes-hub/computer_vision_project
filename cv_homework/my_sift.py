#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正


from sift_pyramid import generate_base_image, compute_number_of_octaves, generate_gaussian_kernels, \
    generate_gaussian_images, generate_DoG_images
from sift_find_scale_space_extrema import find_scale_space_extrema
from sift_remove_duplicate import remove_duplicate_keypoints
from sift_convert_keypoints_to_image_size import convert_keypoints_to_input_image_size
from sift_descriptors import generate_descriptors


#########
# 主函数 #
#########

def compute_keypoints_and_descriptors(
        image,
        sigma=1.6,
        num_intervals=3,
        assumed_blur=0.5,
        image_border_width=5):
    """
    对一张输入图像计算SIFT特征点及其描述子。
    """
    image = image.astype('float32')
    base_image = generate_base_image(image, sigma, assumed_blur)
    num_octaves = compute_number_of_octaves(base_image.shape)
    gaussian_kernels = generate_gaussian_kernels(sigma, num_intervals)
    gaussian_images = generate_gaussian_images(
        base_image, num_octaves, gaussian_kernels)
    dog_images = generate_DoG_images(gaussian_images)
    keypoints = find_scale_space_extrema(
        gaussian_images,
        dog_images,
        num_intervals,
        sigma,
        image_border_width)
    keypoints = remove_duplicate_keypoints(keypoints)
    keypoints = convert_keypoints_to_input_image_size(keypoints)
    descriptors = generate_descriptors(keypoints, gaussian_images)
    return keypoints, descriptors
