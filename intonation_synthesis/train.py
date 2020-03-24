import datetime
import json
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt
from multiprocessing import cpu_count

import const
import settings
from datasets import HTSDataset
from net import build_net
from utils import hprint, pad_array


# tf.compat.v1.disable_eager_execution()

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

CPU_COUNT = \
    int(hyperparameters.get('cpu_count')) \
    if hyperparameters.get('cpu_count') is not None \
    else round(cpu_count() / 2)

GPU_COUNT = \
    int(hyperparameters.get('gpu_count')) \
    if hyperparameters.get('gpu_count') is not None \
    else 1


def pad_data(features, target, max_seq_len=const.MAX_LEN):
    X = numpy.ndarray([max_seq_len, const.FEATURES_ORDER])
    y = numpy.ndarray([max_seq_len, 1])
    features_padded = pad_array(X.shape, features)
    target_padded = pad_array(y.shape, target)

    return features_padded, target_padded


def test(net=None, test_files=None):
    hprint("Testing...")

    hts_testset = HTSDataset(
        settings.WORKDIR + 'input/data/training', file_list=test_files,
        transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
        min_f0=const.MIN_F0, max_f0=const.MAX_F0, batch_size=BATCH_SIZE,
        min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
        rich_feats=const.RICH_FEATS)

    for X_batch, y_batch in hts_testset:
        predictions = net.predict(X_batch, batch_size=BATCH_SIZE)
        for sample_no in range(len(X_batch)):
            f = plt.figure()
            plt.plot(
                predictions[sample_no, :].tolist(), 'r-',
                y_batch[sample_no, :].tolist(), 'b-')
            f.savefig(
                settings.WORKDIR +
                "model/pred_vs_real_sample_{}_{}.pdf".format(
                    sample_no,
                    datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
                bbox_inches='tight')
            plt.close('all')
    with open(settings.WORKDIR + "model/training_report.txt", "w") as report:
        report.writelines("\n".join(test_files))


def train():

    hts_dataset = HTSDataset(
        settings.WORKDIR + 'input/data/training', transform=pad_data,
        max_size=DATASET_SIZE_LIMIT, batch_size=BATCH_SIZE,
        f0_backward_window_len=const.F0_WINDOW_LEN, min_f0=const.MIN_F0,
        max_f0=const.MAX_F0, min_duration=const.MIN_DURATION,
        max_duration=const.MAX_DURATION, rich_feats=const.RICH_FEATS)

    hts_validation_dataset = HTSDataset(
        settings.WORKDIR + 'input/data/training',
        file_list=hts_dataset.validation_data, batch_size=BATCH_SIZE,
        transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
        min_f0=const.MIN_F0, max_f0=const.MAX_F0,
        min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
        rich_feats=const.RICH_FEATS)

    hprint('Dataset size: {}'.format(str(len(hts_dataset))))

    model = build_net(HIDDEN_SIZE, NUM_LAYERS, DROPOUT, gpus=GPU_COUNT)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        settings.WORKDIR + "model/model.hdf5", monitor='val_loss',
        verbose=1, save_best_only=True, mode='auto')
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
        mode='auto', restore_best_weights=True)

    if CPU_COUNT > 0:
        loss_history = model.fit_generator(
            hts_dataset, validation_data=hts_validation_dataset, epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stopping_callback],
            workers=CPU_COUNT, use_multiprocessing=False,
            verbose=2)
    else:
        loss_history = model.fit_generator(
            hts_dataset, validation_data=hts_validation_dataset, epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stopping_callback],
            verbose=2)

    # model.trainable = False
    test(model, hts_dataset.test_data)

    return model


if __name__ == '__main__':
    train()
