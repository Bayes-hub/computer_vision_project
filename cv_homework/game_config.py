# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'game_config.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GameConfig(object):
    def setupUi(self, GameConfig):
        GameConfig.setObjectName("GameConfig")
        GameConfig.resize(639, 482)
        self.centralwidget = QtWidgets.QWidget(GameConfig)
        self.centralwidget.setObjectName("centralwidget")
        self.marker_window = QtWidgets.QLabel(self.centralwidget)
        self.marker_window.setGeometry(QtCore.QRect(30, 10, 271, 311))
        self.marker_window.setText("")
        self.marker_window.setObjectName("marker_window")
        self.open_marker_label = QtWidgets.QLabel(self.centralwidget)
        self.open_marker_label.setGeometry(QtCore.QRect(20, 340, 134, 40))
        self.open_marker_label.setMaximumSize(QtCore.QSize(16777215, 40))
        self.open_marker_label.setStyleSheet("font: 9pt \"黑体\";")
        self.open_marker_label.setObjectName("open_marker_label")
        self.marker_path_button = QtWidgets.QPushButton(self.centralwidget)
        self.marker_path_button.setGeometry(QtCore.QRect(170, 340, 133, 40))
        self.marker_path_button.setMaximumSize(QtCore.QSize(16777215, 40))
        self.marker_path_button.setObjectName("marker_path_button")
        self.marker_img_path = QtWidgets.QLineEdit(self.centralwidget)
        self.marker_img_path.setGeometry(QtCore.QRect(20, 390, 281, 30))
        self.marker_img_path.setMaximumSize(QtCore.QSize(16777215, 30))
        self.marker_img_path.setObjectName("marker_img_path")
        self.instruction_label = QtWidgets.QLabel(self.centralwidget)
        self.instruction_label.setGeometry(QtCore.QRect(321, 14, 301, 261))
        self.instruction_label.setStyleSheet("font: 22pt \"Times New Roman\";")
        self.instruction_label.setObjectName("instruction_label")
        self.start_game_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_game_button.setGeometry(QtCore.QRect(390, 290, 171, 61))
        self.start_game_button.setStyleSheet("font: 16pt \"Times New Roman\";")
        self.start_game_button.setObjectName("start_game_button")
        GameConfig.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GameConfig)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 639, 26))
        self.menubar.setObjectName("menubar")
        GameConfig.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GameConfig)
        self.statusbar.setObjectName("statusbar")
        GameConfig.setStatusBar(self.statusbar)

        self.retranslateUi(GameConfig)
        QtCore.QMetaObject.connectSlotsByName(GameConfig)

    def retranslateUi(self, GameConfig):
        _translate = QtCore.QCoreApplication.translate
        GameConfig.setWindowTitle(_translate("GameConfig", "人机交互游戏《外星人入侵》配置窗口"))
        self.open_marker_label.setText(_translate("GameConfig", "打开标识图像："))
        self.marker_path_button.setText(_translate("GameConfig", "..."))
        self.instruction_label.setText(_translate("GameConfig", "请将标识对准相机\n"
"直至游戏界面弹出"))
        self.start_game_button.setText(_translate("GameConfig", "开始配置"))

