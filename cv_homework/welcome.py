# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'welcome.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_WelcomeWindow(object):
    def setupUi(self, WelcomeWindow):
        WelcomeWindow.setObjectName("WelcomeWindow")
        WelcomeWindow.resize(1038, 697)
        self.centralwidget = QtWidgets.QWidget(WelcomeWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(810, 10, 220, 174))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("220px-Hustseals.png"))
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.specific_object_tracking_button = QtWidgets.QPushButton(self.centralwidget)
        self.specific_object_tracking_button.setGeometry(QtCore.QRect(180, 500, 281, 81))
        self.specific_object_tracking_button.setStyleSheet("font: 12pt \"宋体\";")
        self.specific_object_tracking_button.setObjectName("specific_object_tracking_button")
        self.HCI_game_button = QtWidgets.QPushButton(self.centralwidget)
        self.HCI_game_button.setGeometry(QtCore.QRect(590, 500, 281, 81))
        self.HCI_game_button.setStyleSheet("font: 12pt \"宋体\";")
        self.HCI_game_button.setObjectName("HCI_game_button")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 190, 1013, 68))
        self.label_2.setStyleSheet("font: 36pt \"Times New Roman\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(330, 100, 387, 43))
        self.label_3.setStyleSheet("font: 26pt \"宋体\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(270, 310, 509, 94))
        self.label_4.setStyleSheet("font: 24pt \"Times New Roman\";")
        self.label_4.setObjectName("label_4")
        WelcomeWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(WelcomeWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1038, 26))
        self.menubar.setObjectName("menubar")
        WelcomeWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(WelcomeWindow)
        self.statusbar.setObjectName("statusbar")
        WelcomeWindow.setStatusBar(self.statusbar)

        self.retranslateUi(WelcomeWindow)
        QtCore.QMetaObject.connectSlotsByName(WelcomeWindow)

    def retranslateUi(self, WelcomeWindow):
        _translate = QtCore.QCoreApplication.translate
        WelcomeWindow.setWindowTitle(_translate("WelcomeWindow", "基于SIFT和KCF的运动目标匹配与跟踪"))
        self.specific_object_tracking_button.setText(_translate("WelcomeWindow", "特定目标追踪"))
        self.HCI_game_button.setText(_translate("WelcomeWindow", "人机交互游戏《外星人入侵》"))
        self.label_2.setText(_translate("WelcomeWindow", "基于SIFT和KCF的运动目标匹配与跟踪"))
        self.label_3.setText(_translate("WelcomeWindow", "计算机视觉课程设计"))
        self.label_4.setText(_translate("WelcomeWindow", "李星毅 U201712072 自实1701\n"
"曾文正 U201715853 校交1701"))

