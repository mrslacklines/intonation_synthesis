import datetime
import json
import numpy
import os
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from mxnet import gluon

import const
import settings
from datasets import HTSDataset
from net import build_net
from utils import hprint, pad_array


with open(
        settings.WORKDIR + '/input/config/hyperparameters.json') as json_file:
    hyperparameters = json.load(json_file)

DATASET_SIZE_LIMIT = \
    int(hyperparameters.get('dataset_size_limit')) \
    if hyperparameters.get('dataset_size_limit') is not None else None
BATCH_SIZE = int(hyperparameters.get('batch_size'))
EPOCHS = int(hyperparameters.get('epochs'))

HIDDEN_SIZE = int(hyperparameters.get('hidden_size'))
NUM_LAYERS = int(hyperparameters.get('num_layers'))
DROPOUT = float(hyperparameters.get('dropout'))
# Currently not used (ADAM)
# LEARNING_RATE = float(hyperparameters.get('learning_rate'))

CPU_COUNT = \
    int(hyperparameters.get('cpu_count')) \
    if hyperparameters.get('cpu_count') is not None \
    else round(cpu_count() / 2)


def training_report(loss):
    f = plt.figure()
    plt.plot(loss, 'b-')
    f.savefig(
        settings.WORKDIR +
        "model/loss_{}.pdf".format(
            datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
        bbox_inches='tight')
    plt.close('all')


def pad_data(features, target, max_seq_len=const.MAX_LEN):
    X = numpy.ndarray([max_seq_len, const.FEATURES_ORDER])
    y = numpy.ndarray([max_seq_len, 1])
    features_padded = pad_array(X.shape, features)
    target_padded = pad_array(y.shape, target)

    return features_padded, target_padded


def save(
        net, epoch, loss, holdout_data=None, timestamp_files=False,
        model_dir="/model/"):
    if timestamp_files:
        model_filename = settings.WORKDIR + model_dir + "{}.model".format(
            datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p"))
    else:
        model_filename = settings.WORKDIR + model_dir + "model.model"
    with open("training_report.txt", "a+") as report_file:
        json.dump(
            {
                'epoch': epoch,
                'loss': loss,
                'holdout_data': holdout_data,
            }, report_file)

    net.save(model_filename)


def test(net=None, test_files=None):
    hprint("Testing...")

    hts_testset = HTSDataset(
        settings.WORKDIR + 'input/data/training', file_list=test_files,
        transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
        min_f0=const.MIN_F0, max_f0=const.MAX_F0,
        min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
        rich_feats=const.RICH_FEATS)

    test_data = gluon.data.DataLoader(
        hts_testset, batch_size=BATCH_SIZE, last_batch='discard',
        num_workers=CPU_COUNT)

    for X_batch, y_batch in test_data:
        predictions = net.predict_on_batch(X_batch.asnumpy())
        for sample_no in range(len(X_batch)):
            f = plt.figure()
            plt.plot(
                predictions[sample_no, :].numpy().tolist(), 'r-',
                y_batch[sample_no, :].astype(
                    'float32').asnumpy().tolist(), 'b-')
            f.savefig(
                settings.WORKDIR +
                "model/pred_vs_real_sample_{}_{}.pdf".format(
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
        max_size=DATASET_SIZE_LIMIT,
        f0_backward_window_len=const.F0_WINDOW_LEN, min_f0=const.MIN_F0,
        max_f0=const.MAX_F0, min_duration=const.MIN_DURATION,
        max_duration=const.MAX_DURATION, rich_feats=const.RICH_FEATS)

    hts_validation_dataset = HTSDataset(
        settings.WORKDIR + 'input/data/training',
        file_list=hts_dataset.validation_data,
        transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
        min_f0=const.MIN_F0, max_f0=const.MAX_F0,
        min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
        rich_feats=const.RICH_FEATS)

    hprint('Dataset size: {}'.format(str(len(hts_dataset))))

    train_data = gluon.data.DataLoader(
        hts_dataset, batch_size=BATCH_SIZE,
        num_workers=CPU_COUNT, last_batch='discard')

    validation_data = gluon.data.DataLoader(
        hts_validation_dataset, batch_size=BATCH_SIZE,
        num_workers=CPU_COUNT, last_batch='discard')

    model = build_net(HIDDEN_SIZE, NUM_LAYERS, DROPOUT)

    loss_history = []

    for e in range(EPOCHS):
        hprint("Epoch: {}".format(e))
        for batch_no, (X_batch, y_batch) in enumerate(train_data):
            history = model.fit(
                X_batch.asnumpy(), y_batch.asnumpy(), batch_size=BATCH_SIZE)
        loss_history.extend(history.history['loss'])
        training_report(loss_history)

        save(
            model, epoch=e, loss=history.history['loss'][-1],
            holdout_data=hts_dataset.holdout_data)

    test(model, hts_dataset.test_data)

    return model


if __name__ == '__main__':
    train()
