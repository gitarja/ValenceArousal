from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
import sys
from VideoDemonstration.Training.MainWindow import Ui_MainWindow
from Libs.Utils import timeToInt


class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        estimation_results_file = "D:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\Training\\A3_TS103_20200617_141440_279\\results_timeseries.csv"
        video_file = "D:\\usr\\nishihara\\data\\Yamaha-Experiment (Video)\\200617_GroupA\\TS103\\A3_TS103_logbridge3.mp4"
        time_start = "2020/6/17  14:52:25"
        time_end = "2020/6/17  14:54:20"
        time_start = timeToInt(time_start)
        time_end = timeToInt(time_end)
        self.view = Ui_MainWindow(video_file=video_file, results_file=estimation_results_file, time_start=time_start, time_end=time_end)




    def run(self):
        self.view.show()
        return self.app.exec_()