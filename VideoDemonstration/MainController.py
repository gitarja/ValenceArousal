from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
import sys
from VideoDemonstration.MainWindow import Ui_MainWindow
class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        estimation_results_file = "D:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-10-27\\A6-2020-10-27\\video_results_ecg.csv"
        video_file = "D:\\usr\\nishihara\\GitHub\\ValenceArousal\\VideoDemonstration\\Videos\\N1.mp4"
        self.view = Ui_MainWindow(video_file=video_file, results_file=estimation_results_file, video_idx=2)




    def run(self):
        self.view.show()
        return self.app.exec_()