from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Concatenate, Multiply, ELU


class PercolativeLearning():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def createPercNet(self, input_main, input_aux, alpha):
        x_main = input_main
        x_aux = Multiply()([input_aux, alpha])
        x = Concatenate()([x_main, x_aux])

        for u in [1024, 1024, 512, 512]:
            x = Dense(units=u)(x)
            x = BatchNormalization()(x)
            x = ELU()(x)
            # x = Dropout(0.5)(x)

        return x

    def createIntNet(self, input):
        x = input
        for u in [512, 512, 512]:
            x = Dense(units=u)(x)
            # x = BatchNormalization()(x)
            x = ELU()(x)
            x = Dropout(0.5)(x)

        logit = Dense(units=self.num_classes)(x)
        return logit
