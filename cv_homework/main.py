#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
from utils import find_target
import my_kcf
import numpy as np
from alien_invasion import run_game
from game_config import Ui_GameConfig
from tracking import Ui_TrackingWindow
from welcome import Ui_WelcomeWindow


class WelcomeGUI(QMainWindow, Ui_WelcomeWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


class ObjectTrackingGUI(QMainWindow, Ui_TrackingWindow):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        self.setupUi(self)
        self.call_back_functions()

    def call_back_functions(self):
        self.img_path_button.clicked.connect(self.open_image)
        self.video_path_button.clicked.connect(self.open_video)
        self.start_button.clicked.connect(self.timeout_func)
        self.exit_button.clicked.connect(self.close_window)

    def timeout_func(self):
        self.timer.start(30)
        self.timer.timeout.connect(self.display_frame)

    def display_frame(self):
        if self.cap.isOpened() == False:
            self.cap = cv2.VideoCapture(self.video_path)
        flag, img = self.cap.read()
        if flag:
            boundingbox = self.tracker.update(img)
            # status, boundingbox = self.tracker.update(img)
            boundingbox = list(map(int, boundingbox))
            cv2.rectangle(
                img,
                (boundingbox[0],
                 boundingbox[1]),
                (boundingbox[0] +
                 boundingbox[2],
                 boundingbox[1] +
                 boundingbox[3]),
                (0,
                 255,
                 255),
                5)
            img = QImage(
                img.data,
                img.shape[1],
                img.shape[0],
                QImage.Format_RGB888).rgbSwapped()
            self.video_window.setPixmap(
                QPixmap.fromImage(img).scaled(
                    self.video_window.width(),
                    self.video_window.height()))
        else:
            self.cap.release()
            self.timer.stop()

    def open_image(self):
        img_name, img_type = QFileDialog.getOpenFileName(
            self, "打开图片", "", "All Files(*)")
        self.target_img = img_name.encode('utf-8')
        self.target_img_path.setText(img_name)
        img = QPixmap(img_name).scaled(
            self.target_window.width(),
            self.target_window.height())
        self.target_window.setPixmap(img)

    def open_video(self):
        video_name, video_type = QFileDialog.getOpenFileName(
            self, "打开视频", "", "All Files(*)")
        self.video_path = video_name
        self.test_video_path.setText(video_name)
        self.cap = cv2.VideoCapture(video_name)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if self.cap.isOpened():
            flag, img = self.cap.read()
            self.kcf_init_with_sift(img)
            img = QImage(
                img.data,
                img.shape[1],
                img.shape[0],
                QImage.Format_RGB888).rgbSwapped()
            self.video_window.setPixmap(
                QPixmap.fromImage(img).scaled(
                    self.video_window.width(),
                    self.video_window.height()))

        self.cap = cv2.VideoCapture(self.video_path)

    def kcf_init_with_sift(self, frame):
        target = cv2.imdecode(
            np.fromfile(
                self.target_img,
                dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE)
        init_rect = find_target(target, frame)
        ix = init_rect[0][0]
        iy = init_rect[0][1]
        w = init_rect[3][0] - init_rect[0][0]
        h = init_rect[1][1] - init_rect[0][1]
        self.tracker = my_kcf.KCF(
            True, True, True)
        # self.tracker = cv2.TrackerKCF_create()
        self.tracker.init([ix, iy, w, h], frame)
        # self.tracker.init(frame, (ix, iy, w, h))

    def close_window(self):
        self.cap.release()
        self.timer.stop()
        self.close()


class GameConfigGUI(QMainWindow, Ui_GameConfig):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.call_back_functions()

    def call_back_functions(self):
        self.marker_path_button.clicked.connect(self.open_marker)
        self.start_game_button.clicked.connect(self.start_game)

    def open_marker(self):
        marker_name, marker_type = QFileDialog.getOpenFileName(
            self, "打开图片", "", "All Files(*)")
        self.marker_img = marker_name.encode('utf-8')
        self.marker_img_path.setText(marker_name)
        marker = QPixmap(marker_name).scaled(
            self.marker_window.width(),
            self.marker_window.height())
        self.marker_window.setPixmap(marker)

    def start_game(self):
        run_game(self.marker_img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome = WelcomeGUI()
    object_tracking = ObjectTrackingGUI()
    game_config = GameConfigGUI()

    welcome.specific_object_tracking_button.clicked.connect(
        object_tracking.show)
    welcome.HCI_game_button.clicked.connect(game_config.show)

    welcome.show()
    sys.exit(app.exec_())
