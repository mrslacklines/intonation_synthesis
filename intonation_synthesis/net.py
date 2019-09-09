import mxnet as mx

import const


class Net(mx.gluon.HybridBlock):
    def __init__(
            self, hidden_size, num_layers, dropout, bidirectional, batch_size,
            model_ctx, **kwargs):

        super(Net, self).__init__(**kwargs)

        self.model_ctx = model_ctx
        self.batch_size = batch_size

        self.lstm = mx.gluon.rnn.LSTM(
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, bidirectional=bidirectional, layout='NTC')
        self.output_layer = mx.gluon.nn.Dense(const.MAX_LEN, flatten=True)

    def forward(self, batch):
        dtype = 'float32'
        states = self.lstm.begin_state(
            self.batch_size, ctx=self.model_ctx, dtype=dtype)
        output, _ = self.lstm(batch, states)
        output = self.output_layer(output)
        return output


def build_net(
        hidden_size, num_layers, dropout, bidirectional, batch_size, model_ctx,
        net_type='cnn'):

    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        if net_type == 'cnn':
            for layer_no in range(num_layers):
                net.add(mx.gluon.nn.Conv1D(
                    channels=128, kernel_size=5, activation='relu'))
                net.add(mx.gluon.nn.Conv1D(
                    channels=128, kernel_size=5, activation='relu'))
                net.add(mx.gluon.nn.Dropout(dropout))
                net.add(mx.gluon.nn.MaxPool1D(
                    pool_size=2, strides=2))
                # The Flatten layer collapses all axis,
                # except the first one, into one axis.
                net.add(mx.gluon.nn.Flatten())
                net.add(mx.gluon.nn.Dense(hidden_size, activation="relu"))
            net.add(mx.gluon.nn.Dense(const.MAX_LEN))
        elif net_type == 'lstm':
            # NTC: bach size, sequence length, input size
            net.add(mx.gluon.rnn.LSTM(
                hidden_size=hidden_size, num_layers=num_layers,
                dropout=dropout, bidirectional=bidirectional,
                layout='NTC'))
            net.add(mx.gluon.nn.Dense(const.MAX_LEN, flatten=True))

    return net
