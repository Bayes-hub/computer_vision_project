#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

from cv2 import KeyPoint
from numpy.linalg import det, lstsq
from numpy import all, array, dot, stack, trace, floor, round, float32
from sift_keypoints_orient import compute_keypoints_with_orientations

###################
# 尺度空间上极值搜索 #
###################


def find_scale_space_extrema(
        gaussian_images,
        dog_images,
        num_intervals,
        sigma,
        image_border_width,
        contrast_threshold=0.04):
    """
    尺度空间上的极值搜索。
    """
    threshold = floor(
        0.5 *
        contrast_threshold /
        num_intervals *
        255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(
                zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(
                    image_border_width,
                    first_image.shape[0] -
                    image_border_width):
                for j in range(
                        image_border_width,
                        first_image.shape[1] -
                        image_border_width):
                    if is_pixel_an_extremum(first_image[i -
                                                     1:i +
                                                     2, j -
                                                     1:j +
                                                     2], second_image[i -
                                                                      1:i +
                                                                      2, j -
                                                                      1:j +
                                                                      2], third_image[i -
                                                                                      1:i +
                                                                                      2, j -
                                                                                      1:j +
                                                                                      2], threshold):
                        localization_result = localize_extremum_via_quadratic_fit(
                            i,
                            j,
                            image_index + 1,
                            octave_index,
                            num_intervals,
                            dog_images_in_octave,
                            sigma,
                            contrast_threshold,
                            image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = compute_keypoints_with_orientations(
                                keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


def is_pixel_an_extremum(
        first_subimage,
        second_subimage,
        third_subimage,
        threshold):
    """
    检查中心元素是否严格大于或小于它的3*3*3邻居，若是，返回True，否则返回False。
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= first_subimage) and \
                all(center_pixel_value >= third_subimage) and \
                all(center_pixel_value >= second_subimage[0, :]) and \
                all(center_pixel_value >= second_subimage[2, :]) and \
                center_pixel_value >= second_subimage[1, 0] and \
                center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
                all(center_pixel_value <= third_subimage) and \
                all(center_pixel_value <= second_subimage[0, :]) and \
                all(center_pixel_value <= second_subimage[2, :]) and \
                center_pixel_value <= second_subimage[1, 0] and \
                center_pixel_value <= second_subimage[1, 2]
    return False


def localize_extremum_via_quadratic_fit(
        i,
        j,
        image_index,
        octave_index,
        num_intervals,
        dog_images_in_octave,
        sigma,
        contrast_threshold,
        image_border_width,
        eigenvalue_ratio=10,
        num_attempts_until_convergence=5):
    """
    利用泰勒展开对像素进行亚像素（子像素）精确定位。
    """
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need
        # to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[
            image_index - 1:image_index + 2]
        pixel_cube = stack([first_image[i - 1:i + 2, j - 1:j + 2],
                            second_image[i - 1:i + 2, j - 1:j + 2],
                            third_image[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
        gradient = compute_gradient_at_center_pixel(pixel_cube)
        hessian = compute_hessian_at_center_pixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(
            extremum_update[0]) < 0.5 and abs(
            extremum_update[1]) < 0.5 and abs(
                extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= \
                image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1,
                                                1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * \
            num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                (eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint
            # object
            keypoint = KeyPoint()
            keypoint.pt = ((j +
                            extremum_update[0]) *
                           (2 ** octave_index), (i +
                                                 extremum_update[1]) *
                           (2 ** octave_index))
            keypoint.octave = octave_index + image_index * \
                (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (
                2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None


def compute_gradient_at_center_pixel(pixel_array):
    """
    Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size.
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array
    # axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])


def compute_hessian_at_center_pixel(pixel_array):
    """
    Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size.
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array
    # axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1,
                              2,
                              2] - pixel_array[1,
                                               2,
                                               0] - pixel_array[1,
                                                                0,
                                                                2] + pixel_array[1,
                                                                                 0,
                                                                                 0])
    dxs = 0.25 * (pixel_array[2,
                              1,
                              2] - pixel_array[2,
                                               1,
                                               0] - pixel_array[0,
                                                                1,
                                                                2] + pixel_array[0,
                                                                                 1,
                                                                                 0])
    dys = 0.25 * (pixel_array[2,
                              2,
                              1] - pixel_array[2,
                                               0,
                                               1] - pixel_array[0,
                                                                2,
                                                                1] + pixel_array[0,
                                                                                 0,
                                                                                 1])
    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
