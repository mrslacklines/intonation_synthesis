import datetime
import json
import mxnet as mx
import numpy
import os
from matplotlib import pyplot as plt
from mxnet import autograd, gluon

import const
import settings
from datasets import HTSDataset
from net import build_net
from utils import hprint, pad_array


if settings.TRAIN_ON_GPU:
    MODEL_CTX = mx.gpu()
else:
    MODEL_CTX = mx.cpu()

with open(
        settings.WORKDIR + '/input/config/hyperparameters.json') as json_file:
    hyperparameters = json.load(json_file)

BATCH_SIZE = int(hyperparameters.get('batch_size'))
EPOCHS = int(hyperparameters.get('epochs'))

HIDDEN_SIZE = int(hyperparameters.get('hidden_size'))
NUM_LAYERS = int(hyperparameters.get('num_layers'))
DROPOUT = float(hyperparameters.get('dropout'))
LEARNING_RATE = float(hyperparameters.get('learning_rate'))
BIDIRECTIONAL = bool(hyperparameters.get('bidirectional'))


def pad_data(features, target, max_seq_len=const.MAX_LEN):
    X = numpy.ndarray([max_seq_len, const.FEATURES_ORDER])
    y = numpy.ndarray([max_seq_len, 1])
    features_padded = pad_array(X.shape, features)
    target_padded = pad_array(y.shape, target)

    return features_padded, target_padded


def save(net, created=False):
    if created:
        model_filename = settings.WORKDIR + '/model/{}.params'.format(
            datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p"))
    else:
        model_filename = 'intonation'

    net.save_parameters(model_filename)


def test(net=None, test_files=None):
    hprint("Testing...")

    if settings.TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu())

    hts_testset = HTSDataset(
        settings.WORKDIR + 'input/data/testing', file_list=test_files,
        transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
        min_f0=const.MIN_F0, max_f0=const.MAX_F0,
        min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
        rich_feats=const.RICH_FEATS)

    test_data = gluon.data.DataLoader(
        hts_testset, batch_size=BATCH_SIZE, last_batch='discard',
        num_workers=settings.CPU_COUNT)

    for X_batch, y_batch in test_data:
        if settings.TRAIN_ON_GPU:
            X_batch = X_batch.as_in_context(mx.gpu())
            y_batch = y_batch.as_in_context(mx.gpu())
        else:
            X_batch = X_batch.as_in_context(mx.cpu())
            y_batch = y_batch.as_in_context(mx.cpu())
        if settings.MULTI_PRECISION:
            X_batch = X_batch.astype('float16', copy=False)
            y_batch = y_batch.astype('float16', copy=False)
        with autograd.predict_mode():
            predictions = net(X_batch)

            for sample_no in range(len(X_batch)):
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


def train():

    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('models'):
        os.makedirs('models')

    hts_dataset = HTSDataset(
        settings.WORKDIR + 'input/data/training', transform=pad_data,
        max_size=settings.DATASET_SIZE_LIMIT,
        f0_backward_window_len=const.F0_WINDOW_LEN, min_f0=const.MIN_F0,
        max_f0=const.MAX_F0, min_duration=const.MIN_DURATION,
        max_duration=const.MAX_DURATION, rich_feats=const.RICH_FEATS)

    hprint('Dataset size: {}'.format(str(len(hts_dataset))))

    train_data = gluon.data.DataLoader(
        hts_dataset, batch_size=BATCH_SIZE,
        num_workers=settings.CPU_COUNT, last_batch='discard')
    net = build_net(
        HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, BATCH_SIZE, MODEL_CTX,
        settings.MULTI_PRECISION)
    net.collect_params().initialize(mx.init.Xavier(), ctx=MODEL_CTX)
    if settings.TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu())
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': LEARNING_RATE, 'multi_precision': True})
    l2loss = mx.gluon.loss.L2Loss()

    hprint("Training... ({} epochs)".format(EPOCHS))

    if settings.DEBUG:
        mean_squared_error = mx.metric.MSE()

    for e in range(EPOCHS):
        for X_batch, y_batch in train_data:
            X_batch.wait_to_read()
            y_batch.wait_to_read()
            if settings.TRAIN_ON_GPU:
                X_batch = X_batch.as_in_context(mx.gpu())
                y_batch = y_batch.as_in_context(mx.gpu())
            else:
                X_batch = X_batch.as_in_context(mx.cpu())
                y_batch = y_batch.as_in_context(mx.cpu())
            if settings.MULTI_PRECISION:
                X_batch = X_batch.astype('float16', copy=False)
                y_batch = y_batch.astype('float16', copy=False)
            with autograd.record():
                output = net(X_batch)
                loss = l2loss(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)

            if settings.DEBUG:
                mean_squared_error.update(
                    labels=output, preds=y_batch[:, :, 0])
                mse = mean_squared_error.get()

        hprint("Epoch: {}".format(e))
        if settings.DEBUG:
            print("Epoch {}, mse: {}".format(e, mse[1]))
            print("Epoch {}, loss: {}".format(e, numpy.mean(loss.asnumpy())))

    save(net)

    return net


if __name__ == '__main__':
    with mx.Context(MODEL_CTX):
        train()
