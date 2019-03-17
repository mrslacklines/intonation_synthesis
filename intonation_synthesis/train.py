import datetime
import gc
import mxnet as mx
import numpy
import os
import time
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from mxnet import nd, autograd, gluon

from datasets import HTSDataset
from utils import pad_array


CPU_COUNT = round(cpu_count() / 2)
WORKDIR = '/data'
BATCH_SIZE = CPU_COUNT * 4
DATASET_SIZE_LIMIT = 400
EPOCHS = 50
LEARNING_RATE = 0.01

NUM_LAYERS = 1
# https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-lstm
HIDDEN_SIZE = 1024
DROPOUT = 0.2
USE_MOVING_WINDOW = True
F0_WINDOW_LEN = 20
BIDIRECTIONAL = True if not USE_MOVING_WINDOW else False
TRAIN_ON_GPU = True
RICH_FEATS = False
# FEATURES_ORDER = 4115 if RICH_FEATS else 1531
FEATURES_ORDER = 1531

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
    net = mx.gluon.nn.Sequential()
    with net.name_scope():
        # NTC: bach size, sequence length, input size
        net.add(mx.gluon.rnn.LSTM(
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional, layout='NTC'))
        net.add(mx.gluon.nn.Dense(MAX_LEN, flatten=True))

    return net


def test(net=None, test_files=None):

    if TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu(0))

    hts_testset = HTSDataset(
        WORKDIR, file_list=test_files, transform=pad_data,
        f0_backward_window_len=F0_WINDOW_LEN, min_f0=MIN_F0, max_f0=MAX_F0,
        min_duration=MIN_DURATION, max_duration=MAX_DURATION,
        rich_feats=RICH_FEATS)

    test_data = gluon.data.DataLoader(
        hts_testset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT,
        pin_memory=True)

    mean_squared_error = mx.metric.MSE()

    pred_list = []
    for X_batch, y_batch in test_data:
        gc.collect()
        if TRAIN_ON_GPU:
            X_batch = X_batch.as_in_context(mx.gpu(0))
            y_batch = y_batch.as_in_context(mx.gpu(0))
        with autograd.predict_mode():
            predictions = net(X_batch)
            try:
                predictions.wait_to_read()
            except Exception:
                predictions.wait_to_read()
            pred_list.append((y_batch, predictions))

    for index, (y, pred) in enumerate(pred_list):
        pred.wait_to_read()
        # mean_squared_error.update(labels=y, preds=pred)
        for sample_no in range(len(X_batch)):
            f = plt.figure()
            plt.plot(
                pred[sample_no, :].asnumpy().tolist(), 'r-',
                y[sample_no, :].asnumpy().tolist(), 'b-')
            f.savefig("plots/pred_vs_real_sample_{}_{}.pdf".format(
                sample_no,
                datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
                bbox_inches='tight')
            plt.close('all')
    print(mean_squared_error.get())


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

    # hprint('Dataset size: {}'.format(str(len(hts_dataset))))

    train_data = gluon.data.DataLoader(
        hts_dataset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT)
    net = build_net()
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    if TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu(0))
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})
    l2loss = mx.gluon.loss.L2Loss()

    # hprint("Training... ({} epochs)".format(EPOCHS))

    # loss_sequence = []
    for e in range(EPOCHS):
        # cumulative_loss = 0
        for X_batch, y_batch in train_data:
            gc.collect()
            if TRAIN_ON_GPU:
                X_batch = X_batch.as_in_context(mx.gpu(0))
                y_batch = y_batch.as_in_context(mx.gpu(0))
            # X_batch.wait_to_read()
            # y_batch.wait_to_read()
            with autograd.record():
                output = net(X_batch)
                loss = l2loss(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)
            # mean_loss = nd.mean(loss).asscalar()
            # cumulative_loss += mean_loss
    # model_filename = 'models/{}.params'.format(
    #     datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p"))
    # net.save_parameters(model_filename)


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

    return net, hts_dataset.test_data


if __name__ == '__main__':
    net, test_data = train()
    gc.collect()
    test(net, test_data)
