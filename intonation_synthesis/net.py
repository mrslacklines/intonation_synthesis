from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import multi_gpu_model
from tcn import TCN

import const


def build_net(hidden_size, num_layers, dropout, gpus=0):

    model = Sequential()
    # model.add(Dense(
    #     const.FEATURES_ORDER, activation='relu',
    #     input_shape=(const.MAX_LEN, const.FEATURES_ORDER)))
    model.add(TCN(
        nb_filters=hidden_size, kernel_size=2, nb_stacks=num_layers,
        dilations=[1, 2, 4, 8, 16, 32], padding='causal',
        use_skip_connections=True, dropout_rate=dropout, return_sequences=True,
        use_batch_norm=True, input_shape=(const.MAX_LEN, const.FEATURES_ORDER))
    )
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(1, activation='relu'))

    if gpus > 1:
        model = multi_gpu_model(
            model, gpus=gpus, cpu_merge=False, cpu_relocation=True)

    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    return model
