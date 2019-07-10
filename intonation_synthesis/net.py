import mxnet as mx

import const
import settings


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

        if settings.MULTI_PRECISION:
            self.lstm.cast('float16')
            self.output_layer.cast('float16')

    def forward(self, batch):
        dtype = 'float16' if settings.MULTI_PRECISION else 'float32'
        states = self.lstm.begin_state(
            self.batch_size, ctx=self.model_ctx, dtype=dtype)
        output, _ = self.lstm(batch, states)
        output = self.output_layer(output)
        return output


def build_net(
        hidden_size, num_layers, dropout, bidirectional, batch_size, model_ctx,
        multi_precision=False):

    if multi_precision:
        net = Net()
        with net.name_scope():
            net.hybridize()
            net.cast('float16')
    else:
        net = mx.gluon.nn.HybridSequential()
        with net.name_scope():
            # NTC: bach size, sequence length, input size
            net.add(mx.gluon.rnn.LSTM(
                hidden_size=hidden_size, num_layers=num_layers,
                dropout=dropout, bidirectional=bidirectional, layout='NTC'))
            net.add(mx.gluon.nn.Dense(const.MAX_LEN, flatten=True))

    return net
