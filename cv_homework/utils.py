#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

import numpy as np
import cv2
from numba import jit
import warnings
from my_sift import compute_keypoints_and_descriptors

warnings.filterwarnings('ignore')

# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07


################
# SIFT related #
################


def find_target(target, frame):
    MIN_MATCH_COUNT = 10

    img1 = target  # queryImage
    img2 = frame  # trainImage
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # kp1, des1 = computeKeypointsAndDescriptors(img1)
    # kp2, des2 = computeKeypointsAndDescriptors(img2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        midpoints = []
        for i in range(dst.shape[0]):
            if i + 1 <= 3:
                j = i + 1
            else:
                j = 0
            midpoints.append((dst[i] + dst[j]) / 2)
        midpoints = np.array(midpoints).reshape(-1, 2)

        dst = np.array([[midpoints[0][0], midpoints[3][1]],
                        [midpoints[0][0], midpoints[1][1]],
                        [midpoints[2][0], midpoints[1][1]],
                        [midpoints[2][0], midpoints[3][1]]])

        dst = np.int32(dst)
    else:
        print(
            "Not enough matches are found - %d/%d" %
            (len(good), MIN_MATCH_COUNT))

    return dst


################
# FHOG related #
################


def get_feature_maps(image, k, mapp):
    kernel = np.array([[-1., 0., 1.]], np.float32)

    height = image.shape[0]
    width = image.shape[1]
    assert (image.ndim == 3 and image.shape[2])
    num_channels = 3  # (1 if image.ndim==2 else image.shape[2])

    sizeX = width // k
    sizeY = height // k
    px = 3 * NUM_SECTOR
    p = px
    string_size = sizeX * p

    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['numFeatures'] = p
    mapp['map'] = np.zeros(
        (mapp['sizeX'] *
         mapp['sizeY'] *
         mapp['numFeatures']),
        np.float32)

    dx = cv2.filter2D(np.float32(image), -1, kernel)
    dy = cv2.filter2D(np.float32(image), -1, kernel.T)

    arg_vector = np.arange(
        NUM_SECTOR + 1).astype(np.float32) * np.pi / NUM_SECTOR
    boundary_x = np.cos(arg_vector)
    boundary_y = np.sin(arg_vector)

    r, alfa = func1(dx, dy, boundary_x, boundary_y, height,
                    width, num_channels)  # with @jit
    # ~0.001s

    nearest = np.ones((k), np.int)
    nearest[0:k // 2] = -1

    w = np.zeros((k, 2), np.float32)
    a_x = np.concatenate(
        (k /
         2 -
         np.arange(
             k /
             2) -
         0.5,
         np.arange(
             k /
             2,
             k) -
         k /
         2 +
         0.5)).astype(
        np.float32)
    b_x = np.concatenate((k / 2 + np.arange(k / 2) + 0.5, -
                          np.arange(k / 2, k) + k / 2 - 0.5 + k)).astype(np.float32)
    w[:, 0] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
    w[:, 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))

    mapp['map'] = func2(
        r,
        alfa,
        nearest,
        w,
        k,
        height,
        width,
        sizeX,
        sizeY,
        p,
        string_size)  # with @jit
    # ~0.001s

    return mapp


def normalize_and_truncate(mapp, alfa):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = NUM_SECTOR
    xp = NUM_SECTOR * 3
    pp = NUM_SECTOR * 12

    idx = np.arange(
        0,
        sizeX * sizeY * mapp['numFeatures'],
        mapp['numFeatures']).reshape(
        (sizeX * sizeY,
         1)) + np.arange(p)
    part_of_norm = np.sum(mapp['map'][idx] ** 2, axis=1)  # ~0.0002s

    sizeX, sizeY = sizeX - 2, sizeY - 2

    new_data = func3(
        part_of_norm,
        mapp['map'],
        sizeX,
        sizeY,
        p,
        xp,
        pp)  # with @jit

    # truncation
    new_data[new_data > alfa] = alfa

    mapp['numFeatures'] = pp
    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['map'] = new_data

    return mapp


def pca_feature_maps(mapp):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = mapp['numFeatures']
    pp = NUM_SECTOR * 3 + 4
    yp = 4
    xp = NUM_SECTOR

    nx = 1.0 / np.sqrt(xp * 2)
    ny = 1.0 / np.sqrt(yp)

    new_data = func4(
        mapp['map'],
        p,
        sizeX,
        sizeY,
        pp,
        yp,
        xp,
        nx,
        ny)  # with @jit
    ###

    mapp['numFeatures'] = pp
    mapp['map'] = new_data

    return mapp


@jit
def func1(dx, dy, boundary_x, boundary_y, height, width, num_channels):
    r = np.zeros((height, width), np.float32)
    alfa = np.zeros((height, width, 2), np.int)

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            c = 0
            x = dx[j, i, c]
            y = dy[j, i, c]
            r[j, i] = np.sqrt(x * x + y * y)

            for ch in range(1, num_channels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = np.sqrt(tx * tx + ty * ty)
                if magnitude > r[j, i]:
                    r[j, i] = magnitude
                    c = ch
                    x = tx
                    y = ty

            mmax = boundary_x[0] * x + boundary_y[0] * y
            maxi = 0

            for kk in range(0, NUM_SECTOR):
                dotProd = boundary_x[kk] * x + boundary_y[kk] * y
                if dotProd > mmax:
                    mmax = dotProd
                    maxi = kk
                elif -dotProd > mmax:
                    mmax = -dotProd
                    maxi = kk + NUM_SECTOR

            alfa[j, i, 0] = maxi % NUM_SECTOR
            alfa[j, i, 1] = maxi
    return r, alfa


@jit
def func2(
        r,
        alfa,
        nearest,
        w,
        k,
        height,
        width,
        sizeX,
        sizeY,
        p,
        string_size):
    mapp = np.zeros((sizeX * sizeY * p), np.float32)
    for i in range(sizeY):
        for j in range(sizeX):
            for ii in range(k):
                for jj in range(k):
                    if (i * k + ii > 0) and (i * k + ii < height - \
                        1) and (j * k + jj > 0) and (j * k + jj < width - 1):
                        mapp[i * string_size + j * p + alfa[k * i + ii, j * k + jj, 0]
                             ] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]
                        mapp[i * string_size + j * p + alfa[k * i + ii, j * k + jj, 1] +
                             NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]
                        if (i + nearest[ii] >= 0) and (i + \
                            nearest[ii] <= sizeY - 1):
                            mapp[(i + nearest[ii]) * string_size + j * p + alfa[k * i + ii,
                                                                                j * k + jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]
                            mapp[(i + nearest[ii]) * string_size + j * p + alfa[k * i + ii, j * k + jj,
                                                                                1] + NUM_SECTOR] += r[
                                k * i + ii, j * k + jj] * \
                                w[ii, 1] * w[jj, 0]
                        if (j + nearest[jj] >= 0) and (j + \
                            nearest[jj] <= sizeX - 1):
                            mapp[i * string_size + (j + nearest[jj]) * p + alfa[k * i + ii, j *
                                                                                k + jj, 0]] += r[
                                k * i + ii, j * k + jj] * \
                                w[ii, 0] * w[jj, 1]
                            mapp[i * string_size + (j + nearest[jj]) * p + alfa[k * i + ii,
                                                                                j * k + jj,
                                                                                1] + NUM_SECTOR] += r[k * i + ii,
                                                                                                      j * k + jj] * w[ii,
                                                                                                                      0] * w[jj,
                                                                                                                             1]
                        if ((i +
                             nearest[ii] >= 0) and (i +
                                                    nearest[ii] <= sizeY -
                                                    1) and (j +
                                                            nearest[jj] >= 0) and (j +
                                                                                   nearest[jj] <= sizeX -
                                                                                   1)):
                            mapp[(i + nearest[ii]) * string_size + (j + nearest[jj]) * p + alfa[k * \
                                  i + ii, j * k + jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]
                            mapp[(i + nearest[ii]) * string_size + (j + nearest[jj]) * p + alfa[k * i + ii,
                                                                                                j * k + jj, 1] + NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]
    return mapp


@jit
def func3(part_of_norm, mappmap, sizeX, sizeY, p, xp, pp):
    new_data = np.zeros((sizeY * sizeX * pp), np.float32)
    for i in range(1, sizeY + 1):
        for j in range(1, sizeX + 1):
            pos1 = i * (sizeX + 2) * xp + j * xp
            pos2 = (i - 1) * sizeX * pp + (j - 1) * pp

            valOfNorm = np.sqrt(part_of_norm[i * (sizeX + 2) + j] +
                                part_of_norm[i * (sizeX + 2) + (j + 1)] +
                                part_of_norm[(i + 1) * (sizeX + 2) + j] +
                                part_of_norm[(i + 1) * (sizeX + 2) + (j + 1)]) + FLT_EPSILON
            new_data[pos2:pos2 + p] = mappmap[pos1:pos1 + p] / valOfNorm
            new_data[pos2 + 4 * p:pos2 + 6 *
                     p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            valOfNorm = np.sqrt(part_of_norm[i * (sizeX + 2) + j] +
                                part_of_norm[i * (sizeX + 2) + (j + 1)] +
                                part_of_norm[(i - 1) * (sizeX + 2) + j] +
                                part_of_norm[(i - 1) * (sizeX + 2) + (j + 1)]) + FLT_EPSILON
            new_data[pos2 + p:pos2 + 2 *
                     p] = mappmap[pos1:pos1 + p] / valOfNorm
            new_data[pos2 + 6 * p:pos2 + 8 *
                     p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            valOfNorm = np.sqrt(part_of_norm[i * (sizeX + 2) + j] +
                                part_of_norm[i * (sizeX + 2) + (j - 1)] +
                                part_of_norm[(i + 1) * (sizeX + 2) + j] +
                                part_of_norm[(i + 1) * (sizeX + 2) + (j - 1)]) + FLT_EPSILON
            new_data[pos2 + 2 * p:pos2 + 3 *
                     p] = mappmap[pos1:pos1 + p] / valOfNorm
            new_data[pos2 + 8 * p:pos2 + 10 *
                     p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            valOfNorm = np.sqrt(part_of_norm[i * (sizeX + 2) + j] +
                                part_of_norm[i * (sizeX + 2) + (j - 1)] +
                                part_of_norm[(i - 1) * (sizeX + 2) + j] +
                                part_of_norm[(i - 1) * (sizeX + 2) + (j - 1)]) + FLT_EPSILON
            new_data[pos2 + 3 * p:pos2 + 4 *
                     p] = mappmap[pos1:pos1 + p] / valOfNorm
            new_data[pos2 + 10 * p:pos2 + 12 *
                     p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm
    return new_data


@jit
def func4(mappmap, p, sizeX, sizeY, pp, yp, xp, nx, ny):
    new_data = np.zeros((sizeX * sizeY * pp), np.float32)
    for i in range(sizeY):
        for j in range(sizeX):
            pos1 = (i * sizeX + j) * p
            pos2 = (i * sizeX + j) * pp

            for jj in range(2 * xp):  # 2*9
                new_data[pos2 + jj] = np.sum(mappmap[pos1 + yp * \
                                             xp + jj: pos1 + 3 * yp * xp + jj: 2 * xp]) * ny
            for jj in range(xp):  # 9
                new_data[pos2 + 2 * xp + \
                    jj] = np.sum(mappmap[pos1 + jj: pos1 + jj + yp * xp: xp]) * ny
            for ii in range(yp):  # 4
                new_data[pos2 + 3 * xp + ii] = np.sum(
                    mappmap[pos1 + yp * xp + ii * xp * 2: pos1 + yp * xp + ii * xp * 2 + 2 * xp]) * nx
    return new_data


###############
# KCF related #
###############


def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE)
                                           if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!


def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]


def complex_multiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complex_division(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] +
                    a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] +
                    a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))
    assert (img.ndim == 2)
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]
                           ] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0],
                                      0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, _limit):
    if rect[0] + rect[2] > _limit[0] + _limit[2]:
        rect[2] = _limit[0] + _limit[2] - rect[0]
    if rect[1] + rect[3] > _limit[1] + _limit[3]:
        rect[3] = _limit[1] + _limit[3] - rect[1]
    if rect[0] < _limit[0]:
        rect[2] -= (_limit[0] - rect[0])
        rect[0] = _limit[0]
    if rect[1] < _limit[1]:
        rect[3] -= (_limit[1] - rect[1])
        rect[1] = _limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


def get_border(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def sub_window(img, window, border_type=cv2.BORDER_CONSTANT):
    cut_window = [x for x in window]
    limit(cut_window, [0, 0, img.shape[1], img.shape[0]])  # modify cut_window
    assert (cut_window[2] > 0 and cut_window[3] > 0)
    border = get_border(window, cut_window)
    res = img[cut_window[1]:cut_window[1] + cut_window[3],
              cut_window[0]:cut_window[0] + cut_window[2]]

    if border != [0, 0, 0, 0]:
        res = cv2.copyMakeBorder(
            res,
            border[1],
            border[3],
            border[0],
            border[2],
            border_type)
    return res
