# 浸透学習の実装


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from KnowledgeDistillation.Models.DistillModel import TeacherModel, StudentModel



# 画像のピクセルのうちshuffle_rate(0~1)の割合のものをシャッフルする関数
def shuffle_pixel(data, shuffle_rate):
    dtsize = data.shape[1]
    dtnum_shuffled = int(dtsize * shuffle_rate)
    id_shuffled = []
    i = 0
    while i < dtnum_shuffled:
        rnd = random.randint(0, dtsize - 1)
        if not rnd in id_shuffled:
            id_shuffled.append(rnd)
            i += 1
    id_shuffled = np.array(id_shuffled)
    id = id_shuffled
    id_shuffled = np.random.permutation(id_shuffled)
    idlist = np.arange(dtsize)
    for j in range(dtnum_shuffled):
        idlist[id[j]] = idlist[id_shuffled[j]]
    data = data[:, idlist]
    return data


def shuffle_datasets(x, y):
    dtsize = len(x)
    id_shuffled = np.arange(dtsize)
    id_shuffled = np.random.permutation(id_shuffled)
    x_out = x[id_shuffled, :]
    y_out = y[id_shuffled, :]
    return x_out, y_out


maindt_size = 784       # 主データのサイズ
subdt_size = 784        # 補助データのサイズ
shuffle_rate = 0.5      # 主データのシャッフル率
percnet_size = 100      # 浸透サブネットの各層の素子数
percfeature_size = 100  # 浸透特徴の個数
intnet_size = 100       # 統合サブネットの各層の素子数
output_size = 10        # 出力データのサイズ
epochs_prior = 200      # 事前学習のエポック数
epochs_perc = 1000      # 浸透学習のエポック数
epochs_adj = 300        # 微調整のエポック数
batch_size = 1024       # バッチサイズ
validation_split = 1 / 7  # 評価に用いるデータの割合
test_split = 1 / 7        # テストに用いるデータの割合
verbose = 2             # 学習進捗の表示モード
decay = 0.05            # 減衰率

# MNISTデータの読み込み
(x_train_aux, y_train), (x_test_aux, y_test) = mnist.load_data()
y_train = to_categorical(y_train, output_size)
y_test = to_categorical(y_test, output_size)
# 各データが0~1の値となるように調整
x_train_aux = x_train_aux.astype('float32') / 255
x_test_aux = x_test_aux.astype('float32') / 255
# 28*28ピクセルのデータを784個のデータに平滑化
x_train_aux = x_train_aux.reshape([len(x_train_aux), subdt_size])
x_test_aux = x_test_aux.reshape([len(x_test_aux), subdt_size])
# TrainデータとTestデータを結合，シャッフルして，それをTrain，Validation，Testデータに分ける．
x_aux = np.concatenate([x_train_aux, x_test_aux], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
x_aux, y = shuffle_datasets(x_aux, y)

x_main = shuffle_pixel(x_aux, shuffle_rate)
