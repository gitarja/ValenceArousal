# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ValenceArousalVideo.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph as pg
from pyqtgraph import  PlotWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import pandas as pd
import time

class VideoPlayer(QThread):
    changePixmap = pyqtSignal(QImage)
    changeEmotion = pyqtSignal(float, float)
    def __init__(self, parent=None, video_file=None, results_file=None, video_idx = 0):
        self.video_file = video_file
        self.results = pd.read_csv(results_file)
        video_start = self.results["Start"][0]
        self.results["Start"] = self.results["Start"] - video_start
        self.results["End"] = self.results["End"] - video_start
        self.results = self.results[self.results["VideoIdx"] == video_idx]
        self.valence = self.results["valence"].values
        self.arousal = self.results["arousal"].values
        super(VideoPlayer, self).__init__(parent)

    def run(self):
        frame_rate = 30
        frame_rate_result = 0.15
        prev_result = 0
        result_idx = 0
        prev = 0
        cap = cv2.VideoCapture(self.video_file)
        while True:

            time_elapsed = time.time() - prev
            time_elapsed_result = time.time() - prev_result
            if time_elapsed_result > 1. / frame_rate_result:
                prev_result = time.time()
                self.changeEmotion.emit(self.valence[result_idx], self.arousal[result_idx])
                result_idx+=1
            if time_elapsed > 1. / frame_rate:
                prev = time.time()
                ret, frame = cap.read()
                if ret:
                    # https://stackoverflow.com/a/55468544/6622587
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(761, 721, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)

class Ui_MainWindow(QMainWindow):

    def __init__(self, parent=None, video_file=None, results_file=None, video_idx = 0):
        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setUpGraph()
        video_player = VideoPlayer(self, video_file, results_file, video_idx)
        video_player.changePixmap.connect(self.setImage)
        video_player.changeEmotion.connect(self.plotValenceArousal)
        video_player.start()

    def setUpGraph(self):
        self.graphWidget.setXRange(-2, 2)
        self.graphWidget.setYRange(-2, 2)
        self.graphWidget.plotItem.setLabels(left="Arousal", bottom="Valence")

        self.graphWidget.plotItem.getAxis('bottom').setPen(color=(200, 200, 100))
        self.graphWidget.plotItem.getAxis('left').setPen(color=(200, 200, 100))
        inf_x = pg.InfiniteLine(movable=False, angle=90,
                               labelOpts={'position': 0.1, 'color': (200, 200, 100), 'fill': (200, 200, 200, 50),
                                          'movable': False})
        inf_y = pg.InfiniteLine(movable=False, angle=180,
                               labelOpts={'position': 0.1, 'color': (200, 200, 100), 'fill': (200, 200, 200, 50),
                                          'movable': False})
        pen = pg.mkPen(color="r")
        self.marker = pg.PlotDataItem([0.], [0.], name="current_emotion", pen=pen, symbol='+', symbolSize=30,
                                 symbolBrush=("r"))

        self.graphWidget.addItem(self.marker)
        self.graphWidget.addItem(inf_x)
        self.graphWidget.addItem(inf_y)

    @pyqtSlot(float, float)
    def plotValenceArousal(self, x, y):
        self.marker.setData([x], [y])



    @pyqtSlot(QImage)
    def setImage(self, image):
        self.videoWidget.setPixmap(QPixmap.fromImage(image))





    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 768)
        MainWindow.setMinimumSize(QtCore.QSize(1024, 768))
        MainWindow.setMaximumSize(QtCore.QSize(1024, 768))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphWidget = PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(770, 0, 251, 721))
        self.graphWidget.setObjectName("graphWidget")
        self.videoWidget = QtWidgets.QLabel(self.centralwidget)
        self.videoWidget.setGeometry(QtCore.QRect(0, 0, 761, 721))
        self.videoWidget.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.videoWidget.setObjectName("videoWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


