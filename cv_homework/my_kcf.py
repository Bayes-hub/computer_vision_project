#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

import numpy as np
import cv2
from utils import fftd, real, complex_division, complex_multiplication, rearrange, sub_window, \
    get_feature_maps, normalize_and_truncate, pca_feature_maps


def sub_pixel_peak(left, center, right):
    """
    利用幅值做差来定位峰值的位置，返回的是需要改变的偏移量大小
    """
    divisor = 2 * center - right - left  # float
    return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor


class KCF:
    def __init__(self, hog=False, fixed_window=True, multi_scale=False):
        self.padding = 2.5  # padding边缘
        self.regularization = 0.0001  # lambda为正则项系数
        self.output_sigma_factor = 0.125  # 参数sigma

        if multi_scale:
            self.template_size = 96  # 参数template size
            self.scale_step = 1.05  # 多尺度参数
            self.scale_weight = 0.96
        elif fixed_window:
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        if hog:  # HOG 特征
            self.interp_factor = 0.012
            self.sigma = 0.6  # 高斯核参数
            self.cell_size = 4  # HOG cell 大小
            self._hog_features = True
        else:  # 选项：不用hog，直接用灰度图作为特征
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hog_features = False

        self._tmpl_sz = [0, 0]  # 参数说明： [width,height]  #[int,int]
        # roi参数说明 [x,y,width,height]  #[float,float,float,float]
        self._roi = [0., 0., 0., 0.]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.  # float
        # 参数说明：numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._alphaf = None
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog:
        # (size_patch[2], size_patch[0]*size_patch[1])
        self._tmpl = None
        # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog:
        # (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None

    def init(self, roi, image):  # roi对应[ix,iy,w,h]
        """
        使用的是第一帧和第一帧检测到的框，用来初始化分类器，其中包括第一次训练
        """
        self._roi = list(map(float, roi))  # map是映射，即把roi中的数都转化为float型，
        assert (roi[2] > 0 and roi[3] > 0)  # 检测该条件是否为true，如果是假则会报错
        # 这步只用初始化的时候做，这步得到了size_patch，就是框的大小
        self._tmpl = self.get_features(image, 1)
        self._prob = self.create_gaussian_peak(
            self.size_patch[0], self.size_patch[1])
        self._alphaf = np.zeros(
            (self.size_patch[0],
             self.size_patch[1],
             2),
            np.float32)  # alphaf是每次训练得到的参数
        self.train(self._tmpl, 1.0)

    def detect(self, z, x):
        """
        用于检测当前帧目标位置，得到框和响应值
        z为前一帧框内特征图，x是当前帧特征图
        套论文中的公式，然后找到响应最大的地方，就是目标
        返回检测框的位置和响应值
        """
        k = self.gaussian_correlation(
            x, z)  # 计算x与z之间的高斯相关核（套公式），也就是原特征图和现特征图的相关
        res = real(
            fftd(
                complex_multiplication(
                    self._alphaf,
                    fftd(k)),
                True))  # a和k的傅里叶域哈达玛积

        # 这是opencv的求矩阵最大最小值及其坐标的函数，pi对应第四个返回参数，是最大值的坐标
        _, pv, _, pi = cv2.minMaxLoc(res)
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]
        # 上面两行得到了峰值位置的像素坐标，是浮点型的，接下来通过插值变为整数
        if 0 < pi[0] < res.shape[1] - 1:
            p[0] += sub_pixel_peak(res[pi[1], pi[0] - 1],
                                   pv, res[pi[1], pi[0] + 1])
        if 0 < pi[1] < res.shape[0] - 1:
            p[1] += sub_pixel_peak(res[pi[1] - 1, pi[0]],
                                   pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv
        # 返回新框的位置和响应值

    def train(self, x, train_interp_factor):
        """
        用当前帧检测的框内图像进行训练，利用岭回归，通过核函数简化运算，
        循环矩阵产生负样本，实际上并没有体现在公式中因为利用性质把运算化简到了极致。
        通过训练得到alphaf和新的特征图用于下一帧的检测
        """
        k = self.gaussian_correlation(x, x)
        alphaf = complex_division(
            self._prob,
            fftd(k) +
            self.regularization)  # 计算岭回归系数（套那个a=(K+lambda)-1y）

        self._tmpl = (1 - train_interp_factor) * self._tmpl + \
            train_interp_factor * x  # 这里就是(1-p)*x(t-1)+p*x(t)
        self._alphaf = (1 - train_interp_factor) * self._alphaf + \
            train_interp_factor * alphaf  # 更新岭回归系数

    def update(self, image):
        """
        基于当前帧更新目标框位置，image参数就是当前帧，该函数的大体流程为：
        用不同的尺度调用detect函数，可以得到三次响应结果，选择最大的那个作为本次检测到的框
        然后把本次框内图像进行训练，先调用getfeature函数提取hog特征然后调用train函数训练，最后返回当前检测框的位置
        """
        # 对边界进行修正
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[2] + 1
        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.
        # 尺度不变时检测峰值结果，调用detect函数实现
        loc, peak_value = self.detect(
            self._tmpl, self.get_features(
                image, 0, 1.0))

        if self.scale_step != 1:
            # 利用多尺度，这里先尝试一个较小的尺度，同样利用detect函数
            new_loc1, new_peak_value1 = self.detect(
                self._tmpl, self.get_features(
                    image, 0, 1.0 / self.scale_step))
            # 这里尝试一个较大尺度
            new_loc2, new_peak_value2 = self.detect(
                self._tmpl, self.get_features(
                    image, 0, self.scale_step))

            # 比较不同尺度得到的响应，选择最大的那个作为最终框
            if self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
                loc = new_loc1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif self.scale_weight * new_peak_value2 > peak_value:
                loc = new_loc2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step

        # 返回的只有中心坐标，使用尺度和中心坐标调整目标框
        self._roi[0] = cx - self._roi[2] / 2.0 + \
            loc[0] * self.cell_size * self._scale
        self._roi[1] = cy - self._roi[3] / 2.0 + \
            loc[1] * self.cell_size * self._scale

        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self.get_features(image, 0, 1.0)  # 更新完这一帧的框后提取特征拿去训练
        self.train(x, self.interp_factor)  # 继续训练

        return self._roi  # 返回框

    def create_hanning_mats(self):
        """
        初始化汉宁窗，汉宁窗是为了减少频谱泄露
        """
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t /
                                   (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t /
                                   (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self._hog_features:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def gaussian_correlation(self, x1, x2):
        """
        计算高斯卷积
        """
        if self._hog_features:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                x1aux = x1[i, :].reshape(
                    (self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape(
                    (self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(
                    fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                c += caux
            c = rearrange(c)
        else:
            c = cv2.mulSpectrums(
                fftd(x1),
                fftd(x2),
                0,
                conjB=True)  # 'conjB=' 是必要的
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) -
                 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif x1.ndim == 2 and x2.ndim == 2:
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    def create_gaussian_peak(self, sizey, sizex):
        """
        创建高斯峰函数，该函数只在第一帧执行
        """
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / \
            self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def get_features(self, image, init_hann, scale_adjust=1.0):
        """
        用来提取hog特征
        """
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float  求框的中心
        cy = self._roi[1] + self._roi[3] / 2  # float

        if init_hann:
            padded_w = self._roi[2] * self.padding  # padding是类开始定义的一个常数，等于2.5
            padded_h = self._roi[3] * self.padding

            if self.template_size > 1:
                if padded_w >= padded_h:
                    # 类的第一个函数设置的，template_size=96
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                # 类的第一个函数设置的， tmpl_sz=[0,0] int
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if self._hog_features:  # //: 向下取接近商的整数，-9//2=-5  9//2=4
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (
                    2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (
                    2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:  # cell_size: 类的第一个函数，hog的cell_size设为4
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = sub_window(image, extracted_roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
            z = cv2.resize(z, tuple(self._tmpl_sz))
        # 提取hog特征
        if self._hog_features:
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = get_feature_maps(z, self.cell_size, mapp)
            mapp = normalize_and_truncate(mapp, 0.2)
            mapp = pca_feature_maps(mapp)
            self.size_patch = list(
                map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
            features_map = mapp['map'].reshape(
                (self.size_patch[0] * self.size_patch[1],
                 self.size_patch[2])).T  # (size_patch[2], size_patch[0]*size_patch[1])
        else:
            if z.ndim == 3 and z.shape[2] == 3:
                # z:(size_patch[0], size_patch[1], 3)
                # features_map:(size_patch[0], size_patch[1])   #np.int8
                # #0~255
                features_map = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            elif z.ndim == 2:
                # (size_patch[0], size_patch[1]) #np.int8  #0~255
                features_map = z
            features_map = features_map.astype(np.float32) / 255.0 - 0.5
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if init_hann:
            self.create_hanning_mats()  # create_hanning_mats need size_patch

        features_map = self.hann * features_map  # hanning窗，低通滤波器将hog特征图进行滤波
        return features_map
