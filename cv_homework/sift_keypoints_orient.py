#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

from cv2 import KeyPoint
from numpy import arctan2, exp, logical_and, roll, sqrt, rad2deg, where, zeros, round, float32

# 全局变量
float_tolerance = 1e-7

###################
# 特征点主方向的计算 #
###################


def compute_keypoints_with_orientations(
        keypoint,
        octave_index,
        gaussian_image,
        radius_factor=3,
        num_bins=36,
        peak_ratio=0.8,
        scale_factor=1.5):
    """
    计算每一个特征点的方向。
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    # compare with keypoint.size computation in
    # localizeExtremumViaQuadraticFit()
    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(
                    round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - \
                        gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - \
                        gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    # constant in front of exponential can be dropped because
                    # we will find peaks later
                    weight = exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(
                        round(
                            gradient_orientation *
                            num_bins /
                            360.))
                    raw_histogram[histogram_index %
                                  num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(
            n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(
        logical_and(
            smooth_histogram > roll(
                smooth_histogram, 1), smooth_histogram > roll(
                smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(
                *keypoint.pt,
                keypoint.size,
                orientation,
                keypoint.response,
                keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations
