#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

from numpy import array

################
# 特征点尺度转换 #
################


def convert_keypoints_to_input_image_size(keypoints):
    """
    Convert keypoint point, size, and octave to input image size.
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (
            keypoint.octave & ~255) | (
            (keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints
