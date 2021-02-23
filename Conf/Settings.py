# ECG Const
FS_ECG = 250
FS_ECG_ROAD = 200
FS_EEG = 1000
FS_GSR = 256
FS_RESP = 256
SPLIT_TIME = 45
STRIDE = 0.2
EXTENTION_TIME = 2.5

# Features
FEATURES_N = 2480
# FEATURES_N = 13
ECG_RAW_N = 9000
# ECG_RAW_N = 11125
EEG_RAW_N = 9000
EEG_RAW_CH = 19
EEG_RAW_MAX = 2.0426152724552216
EEG_RAW_MIN = -2.169352558978187

# features
EDA_N = 1102
PPG_N = 11
Resp_N = 11
ECG_Resp_N = 13
ECG_N = 13
EEG_N = 1102

# Path Const
DATASET_PATH = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\"
RESULTS_PATH = "\\results_stride=" + str(STRIDE) + "\\"
EEG_PATH = RESULTS_PATH + "eeg\\"
EEG_R_PATH = RESULTS_PATH + "EEG_RAW\\"
EDA_PATH = RESULTS_PATH + "eda\\"
PPG_PATH = RESULTS_PATH + "ppg\\"
RESP_PATH = RESULTS_PATH + "resp\\"
ECG_PATH = RESULTS_PATH + "ecg\\"
ECG_R_PATH = RESULTS_PATH + "ECG_RAW\\"
ECG_RR_PATH = RESULTS_PATH + "ECG_RESP_RAW\\"
ECG_RESP_PATH = RESULTS_PATH + "ecg_resp\\"
TENSORBOARD_PATH = "G:\\usr\\nishihara\\result\\EmotionRecognition\\tensorboard\\"

#results raw
RESULTS_RAW_PATH = "\\results_raw\\"
EEG_RAW_PATH = RESULTS_RAW_PATH + "EEG\\"
EDA_RAW_PATH = RESULTS_RAW_PATH + "eda\\"
PPG_RAW_PATH = RESULTS_RAW_PATH + "ppg\\"
RESP_RAW_PATH = RESULTS_RAW_PATH + "Resp\\"
ECG_RAW_PATH = RESULTS_RAW_PATH + "ECG\\"
ECG_RAW_RESP_PATH = RESULTS_RAW_PATH + "ECG_resp\\"


#manager
CHECK_POINT_PATH = "G:\\usr\\nishihara\\result\\EmotionRecognition\\"
TRAINING_RESULTS_PATH = "G:\\usr\\nishihara\\result\\EmotionRecognition\\"

#road test
ROAD_ECG = "D:\\usr\\pras\\data\\Yamaha-Experiment-Filtered\\road_test\\"

#proportion
HIGH_PROP = 0.5
LOW_PROP = 0.3