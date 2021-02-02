# ECG Const
FS_ECG = 250
FS_ECG_ROAD = 200
FS_EEG = 1000
FS_GSR = 256
FS_RESP = 256
SPLIT_TIME = 45
STRIDE = 0.1
EXTENTION_TIME = 2.5

# Features
FEATURES_N = 2480
# FEATURES_N = 13
ECG_RAW_N = 11250
# ECG_RAW_N = 11664

# features
EDA_N = 553
PPG_N = 11
Resp_N = 11
ECG_Resp_N = 13
ECG_N = 13
EEG_N = 912

# Path Const
DATASET_PATH = "D:\\usr\\pras\\data\\Yamaha-Experiment-Filtered\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\"
RESULTS_PATH = "\\results_stride=0.1\\"
EEG_PATH = RESULTS_PATH + "EEG\\"
EDA_PATH = RESULTS_PATH + "eda\\"
PPG_PATH = RESULTS_PATH + "ppg\\"
RESP_PATH = RESULTS_PATH + "Resp\\"
ECG_PATH = RESULTS_PATH + "ECG\\"
ECG_R_PATH = RESULTS_PATH + "ECG_RAW\\"
ECG_RR_PATH = RESULTS_PATH + "ECG_RESP_RAW\\"
ECG_RESP_PATH = RESULTS_PATH + "ECG_resp\\"
TENSORBOARD_PATH = "D:\\usr\\pras\\result\\ValenceArousal\\tensorboard\\sMCL\\"

#results raw
RESULTS_RAW_PATH = "\\results_raw\\"
EEG_RAW_PATH = RESULTS_RAW_PATH + "EEG\\"
EDA_RAW_PATH = RESULTS_RAW_PATH + "eda\\"
PPG_RAW_PATH = RESULTS_RAW_PATH + "ppg\\"
RESP_RAW_PATH = RESULTS_RAW_PATH + "Resp\\"
ECG_RAW_PATH = RESULTS_RAW_PATH + "ECG\\"
ECG_RAW_RESP_PATH = RESULTS_RAW_PATH + "ECG_resp\\"


#manager
CHECK_POINT_PATH = "D:\\usr\\pras\\result\\ValenceArousal\\model\\sMCL\\"
TRAINING_RESULTS_PATH = "D:\\usr\\pras\\result\\ValenceArousal\\"

#road test
ROAD_ECG = "D:\\usr\\pras\\data\\Yamaha-Experiment-Filtered\\road_test\\"