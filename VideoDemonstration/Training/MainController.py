from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
import sys
from VideoDemonstration.Training.MainWindow import Ui_MainWindow
from Libs.Utils import timeToInt


class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        estimation_results_file = "D:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\Training\\A1_TS101_20200617_141244_537\\results_timeseries.csv"
        video_file = "D:\\usr\\nishihara\\GitHub\\ValenceArousal\\VideoDemonstration\\Training\\Videos\\A1_TS101_Eight.mp4"
        time_start = "2020/6/17  14:33:42"
        time_end = "2020/6/17  14:35:50"
        time_start = timeToInt(time_start)
        time_end = timeToInt(time_end)
        self.view = Ui_MainWindow(video_file=video_file, results_file=estimation_results_file, time_start=time_start, time_end=time_end)




    def run(self):
        self.view.show()
        return self.app.exec_()