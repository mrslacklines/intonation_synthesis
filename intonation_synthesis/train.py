import datetime
import gc
import mxnet as mx
import numpy
import os
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from mxnet import autograd, gluon

from datasets import HTSDataset
from utils import hprint, pad_array


DEBUG = False

CPU_COUNT = round(cpu_count() / 2)
WORKDIR = 'data' if DEBUG else '/data'
BATCH_SIZE = 8
DATASET_SIZE_LIMIT = 400
EPOCHS = 200
LEARNING_RATE = 0.01

SAVE_PARAMS = False

NUM_LAYERS = 3
# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-lstm
HIDDEN_SIZE = 256
DROPOUT = 0.2
USE_MOVING_WINDOW = True
F0_WINDOW_LEN = 20
BIDIRECTIONAL = True if not USE_MOVING_WINDOW else False
TRAIN_ON_GPU = False  # if DEBUG else True
MULTI_PRECISION = False
RICH_FEATS = False
FEATURES_ORDER = 4115 if RICH_FEATS else 1531

MAX_DURATION = 14950000.0
MIN_DURATION = 0
MIN_F0 = -20.299814419936542
MAX_F0 = 35.15619563784128
MAX_LEN = 1900

if TRAIN_ON_GPU:
    model_ctx = mx.gpu()
else:
    model_ctx = mx.cpu()


def pad_data(features, target, max_seq_len=MAX_LEN):
    X = numpy.ndarray([max_seq_len, FEATURES_ORDER])
    y = numpy.ndarray([max_seq_len, 1])
    features_padded = pad_array(X.shape, features)
    target_padded = pad_array(y.shape, target)

    return features_padded, target_padded


def build_net(
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL):

    class Net(mx.gluon.HybridBlock):
        def __init__(
                self, hidden_size=hidden_size, num_layers=num_layers,
                dropout=dropout, bidirectional=bidirectional, **kwargs):

            super(Net, self).__init__(**kwargs)

            self.lstm = mx.gluon.rnn.LSTM(
                hidden_size=hidden_size, num_layers=num_layers,
                dropout=dropout, bidirectional=bidirectional, layout='NTC')
            self.output_layer = mx.gluon.nn.Dense(MAX_LEN, flatten=True)

            if MULTI_PRECISION:
                self.lstm.cast('float16')
                self.output_layer.cast('float16')

        def forward(self, batch):
            dtype = 'float16' if MULTI_PRECISION else 'float32'
            states = self.lstm.begin_state(
                BATCH_SIZE, ctx=model_ctx, dtype=dtype)
            output, _ = self.lstm(batch, states)
            # sequence_level = output.max(axis=1)
            output = self.output_layer(output)
            return output

    if MULTI_PRECISION:
        net = Net()
        with net.name_scope():
            net.hybridize()
            net.cast('float16')
    else:
        net = mx.gluon.nn.HybridSequential()
        with net.name_scope():
            # NTC: bach size, sequence length, input size
            net.add(mx.gluon.rnn.LSTM(
                hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                bidirectional=bidirectional, layout='NTC'))
            net.add(mx.gluon.nn.Dense(MAX_LEN, flatten=True))

    return net


def test(net=None, test_files=None):
    hprint("Testing...")

    if TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu())

    hts_testset = HTSDataset(
        WORKDIR, file_list=test_files, transform=pad_data,
        f0_backward_window_len=F0_WINDOW_LEN, min_f0=MIN_F0, max_f0=MAX_F0,
        min_duration=MIN_DURATION, max_duration=MAX_DURATION,
        rich_feats=RICH_FEATS)

    test_data = gluon.data.DataLoader(
        hts_testset, batch_size=BATCH_SIZE, last_batch='discard',
        num_workers=CPU_COUNT)

    # mean_squared_error = mx.metric.MSE()

    # pred_list = []
    for X_batch, y_batch in test_data:
        if TRAIN_ON_GPU:
            X_batch = X_batch.as_in_context(mx.gpu())
            y_batch = y_batch.as_in_context(mx.gpu())
        else:
            X_batch = X_batch.as_in_context(mx.cpu())
            y_batch = y_batch.as_in_context(mx.cpu())
        if MULTI_PRECISION:
            X_batch = X_batch.astype('float16', copy=False)
            y_batch = y_batch.astype('float16', copy=False)
        with autograd.predict_mode():
            predictions = net(X_batch)
            # gc.collect()
            # pred_list.append((y_batch, predictions))
            for sample_no in range(len(X_batch)):
                # gc.collect()
                f = plt.figure()
                plt.plot(
                    predictions[sample_no, :].astype(
                        'float32').asnumpy().tolist(), 'r-',
                    y_batch[sample_no, :].astype(
                        'float32').asnumpy().tolist(), 'b-')
                f.savefig("plots/pred_vs_real_sample_{}_{}.pdf".format(
                    sample_no,
                    datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
                    bbox_inches='tight')
                plt.close('all')

    # for index, (y, pred) in enumerate(pred_list):
        # mean_squared_error.update(labels=y, preds=pred)

    if SAVE_PARAMS:
        model_filename = 'models/{}.params'.format(
            datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p"))
        net.save_parameters(model_filename)
    # hprint(mean_squared_error.get())


def train():

    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('models'):
        os.makedirs('models')

    hts_dataset = HTSDataset(
        WORKDIR, transform=pad_data, max_size=DATASET_SIZE_LIMIT,
        f0_backward_window_len=F0_WINDOW_LEN, min_f0=MIN_F0, max_f0=MAX_F0,
        min_duration=MIN_DURATION, max_duration=MAX_DURATION,
        rich_feats=RICH_FEATS)

    hprint('Dataset size: {}'.format(str(len(hts_dataset))))

    train_data = gluon.data.DataLoader(
        hts_dataset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT,
        last_batch='discard')
    net = build_net()
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    if TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu())
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': LEARNING_RATE, 'multi_precision': True})
    l2loss = mx.gluon.loss.L2Loss()

    hprint("Training... ({} epochs)".format(EPOCHS))

    # loss_sequence = []
    for e in range(EPOCHS):
        print(e)
        # cumulative_loss = 0
        for X_batch, y_batch in train_data:
            X_batch.wait_to_read()
            y_batch.wait_to_read()
            if TRAIN_ON_GPU:
                X_batch = X_batch.as_in_context(mx.gpu())
                y_batch = y_batch.as_in_context(mx.gpu())
            else:
                X_batch = X_batch.as_in_context(mx.cpu())
                y_batch = y_batch.as_in_context(mx.cpu())
            if MULTI_PRECISION:
                X_batch = X_batch.astype('float16', copy=False)
                y_batch = y_batch.astype('float16', copy=False)
            with autograd.record():
                output = net(X_batch)
                loss = l2loss(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)
            # mean_loss = nd.mean(loss).asscalar()
            # cumulative_loss += mean_loss

        # hprint("Epoch {}, loss: {}".format(
        #     e, cumulative_loss / y_batch[0].shape[0]))
        # print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(
        #     None, e))
        # loss_sequence.append(loss)
    # f = plt.figure()
    # plt.plot(loss_sequence)
    # f.savefig("plots/loss_sequence_{}.pdf".format(
    #     datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
    #     bbox_inches='tight')
    # plt.close('all')
    test(net, hts_dataset.test_data)

    # return net, hts_dataset.test_data


if __name__ == '__main__':
    with mx.Context(model_ctx):
        train()
