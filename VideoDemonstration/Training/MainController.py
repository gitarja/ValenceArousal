from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
import sys
from VideoDemonstration.Training.MainWindow import Ui_MainWindow
from Libs.Utils import timeToInt


class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        estimation_results_file = "D:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\Training\\A2_TS102_20200617_141444_642\\results_timeseries.csv"
        video_file = "D:\\usr\\nishihara\\data\\Yamaha-Experiment (Video)\\200617_GroupA\\TS102\\A2_TS102_eight1.mp4"
        time_start = "2020/6/17  14:28:49"
        time_end = "2020/6/17  14:32:50"
        time_start = timeToInt(time_start)
        time_end = timeToInt(time_end)
        self.view = Ui_MainWindow(video_file=video_file, results_file=estimation_results_file, time_start=time_start, time_end=time_end)




    def run(self):
        self.view.show()
        return self.app.exec_()