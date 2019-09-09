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
NET_TYPE = hyperparameters.get('net_type') or 'lstm'


def pad_data(features, target, max_seq_len=const.MAX_LEN):
    X = numpy.ndarray([max_seq_len, const.FEATURES_ORDER])
    y = numpy.ndarray([max_seq_len, 1])
    features_padded = pad_array(X.shape, features)
    target_padded = pad_array(y.shape, target)

    return features_padded, target_padded


def save(net, epoch, model_dir="/model/"):
    model_filename = settings.WORKDIR + model_dir + "{}.params".format(
        datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p"))

    net.export(model_filename, epoch=epoch)


def test(net=None, test_files=None):
    hprint("Testing...")

    if settings.TRAIN_ON_GPU:
        net.collect_params().reset_ctx(settings.MODEL_CTX)

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
        if settings.TRAIN_ON_GPU and settings.GPU_COUNT <= 1:
            X_batch = X_batch.as_in_context(settings.MODEL_CTX)
            y_batch = y_batch.as_in_context(settings.MODEL_CTX)
        elif settings.TRAIN_ON_GPU and settings.GPU_COUNT > 1:
            pass
        else:
            X_batch = X_batch.as_in_context(settings.MODEL_CTX)
            y_batch = y_batch.as_in_context(settings.MODEL_CTX)

        with autograd.predict_mode():
            predictions = net(X_batch)

            for sample_no in range(len(X_batch)):
                f = plt.figure()
                plt.plot(
                    predictions[sample_no, :].astype(
                        'float32').asnumpy().tolist(), 'r-',
                    y_batch[sample_no, :].astype(
                        'float32').asnumpy().tolist(), 'b-')
                f.savefig(
                    settings.WORKDIR +
                    "output/pred_vs_real_sample_{}_{}.pdf".format(
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
        HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, BATCH_SIZE,
        settings.MODEL_CTX, net_type=NET_TYPE)
    net.collect_params().initialize(
        mx.init.Xavier(), force_reinit=True, ctx=settings.MODEL_CTX)

    if settings.TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu())
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': LEARNING_RATE},
        compression_params={'type': '2bit', 'threshold': 0.5})
    l2loss = mx.gluon.loss.L2Loss()

    net.hybridize()

    hprint("Training... ({} epochs)".format(EPOCHS))

    if settings.DEBUG:
        mean_squared_error = mx.metric.MSE()

    loss_history = []
    mse_history = []

    for e in range(EPOCHS):
        hprint("Epoch: {}".format(e))
        total_loss = 0
        if settings.TRAIN_ON_GPU and settings.GPU_COUNT > 1:
            mdevice_loss = 0.0
        for batch_no, (X_batch, y_batch) in enumerate(train_data):
            X_batch.wait_to_read()
            y_batch.wait_to_read()
            if settings.TRAIN_ON_GPU and settings.GPU_COUNT > 1:
                data_list = gluon.utils.split_and_load(
                    X_batch, settings.MODEL_CTX[:BATCH_SIZE])
                label_list = gluon.utils.split_and_load(
                    y_batch, settings.MODEL_CTX[:BATCH_SIZE])
            elif settings.TRAIN_ON_GPU and settings.GPU_COUNT <= 1:
                X_batch = X_batch.as_in_context(settings.MODEL_CTX)
                y_batch = y_batch.as_in_context(settings.MODEL_CTX)
            else:
                X_batch = X_batch.as_in_context(settings.MODEL_CTX)
                y_batch = y_batch.as_in_context(settings.MODEL_CTX)

            if settings.TRAIN_ON_GPU and settings.GPU_COUNT > 1:
                with autograd.record():
                    outputs = [net(X) for X in data_list]
                    losses = [
                        l2loss(output, y) for output, y
                        in zip(outputs, label_list)]
                for l in losses:
                    l.backward()
                for l in losses:
                    l.wait_to_read()
                trainer.step(BATCH_SIZE, ignore_stale_grad=True)
                # sum losses over all devices
                mdevice_loss += sum([l.sum().asscalar() for l in losses])
                total_loss += mx.nd.sum(mdevice_loss).asscalar()

            else:
                with autograd.record():
                    output = net(X_batch)
                    loss = l2loss(output, y_batch)
                loss.backward()
                loss.wait_to_read()
                trainer.step(BATCH_SIZE)

                total_loss += mx.nd.sum(loss).asscalar()

                if settings.DEBUG:
                    mean_squared_error.update(
                        labels=output, preds=y_batch[:, :, 0])
                    mse = mean_squared_error.get()

        current_loss = total_loss / len(train_data) / BATCH_SIZE

        print('Loss = {}; MSE = {};'.format(
            current_loss,
            mse[1]))

        loss_history.append(current_loss)
        mse_history.append(mse[1])

    f = plt.figure()
    plt.plot(loss_history, 'r-')
    f.savefig(
        settings.WORKDIR +
        "output/loss_{}.pdf".format(
            datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
        bbox_inches='tight')
    plt.close('all')

    save(net, e)

    test(net, hts_dataset.test_data)

    return net


if __name__ == '__main__':
    train()
