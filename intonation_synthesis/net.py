from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN

import const
# from harhn import HARHN


def build_net(hidden_size, num_layers, dropout):

    model = Sequential()
    model.add(TCN(
        nb_filters=hidden_size, kernel_size=2, nb_stacks=num_layers,
        dilations=[1, 2, 4, 8, 16, 32], padding='causal',
        use_skip_connections=True, dropout_rate=dropout, return_sequences=True,
        activation='relu', kernel_initializer='glorot_normal',
        use_batch_norm=True, input_shape=(const.MAX_LEN, const.FEATURES_ORDER))
    )
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse')

    return model
