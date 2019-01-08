import datetime
import mxnet as mx
import numpy
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from mxnet import nd, autograd, gluon

from datasets import HTSDataset
from utils import pad_array


CPU_COUNT = cpu_count()
WORKDIR = 'data'
BATCH_SIZE = 2
DATASET_SIZE_LIMIT = 10
EPOCHS = 30
LEARNING_RATE = 0.1

NUM_LAYERS = 3
HIDDEN_SIZE = 20
DROPOUT = 0.2
USE_MOVING_WINDOW = True
F0_WINDOW_LEN = 20
BIDIRECTIONAL = True if not USE_MOVING_WINDOW else False
TRAIN_ON_GPU = False
FEATURES_ORDER = 4115

MAX_DURATION = 14950000.0
MIN_DURATION = 0
MIN_F0 = -20.299814419936542
MAX_F0 = 35.15619563784128
MAX_LEN = 1900


model_ctx = mx.cpu()


def hprint(string):
    print('=' * 50)
    print(string)
    print('=' * 50)


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
        net.add(mx.gluon.rnn.LSTM(
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional, layout='NTC'))
        net.add(mx.gluon.nn.Dense(MAX_LEN, flatten=True))

    return net


def test(net=None, test_files=None):

    hts_testset = HTSDataset(
        WORKDIR, file_list=test_files, transform=pad_data,
        f0_backward_window_len=F0_WINDOW_LEN, min_f0=MIN_F0, max_f0=MAX_F0,
        min_duration=MIN_DURATION, max_duration=MAX_DURATION)

    test_data = gluon.data.DataLoader(
        hts_testset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT)

    pred_list = []
    for X_batch, y_batch in test_data:
        with autograd.predict_mode():
            predictions = net(X_batch)
            pred_list.append((y_batch, predictions))

    for index, (y, pred) in enumerate(pred_list):
        for sample_no in range(BATCH_SIZE):
            f = plt.figure()
            plt.plot(
                pred[sample_no, :].asnumpy().tolist(), 'r-',
                y[sample_no, :].asnumpy().tolist(), 'b-')
            f.savefig("plots/pred_vs_real_sample_{}_{}.pdf".format(
                sample_no,
                datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
                bbox_inches='tight')


def train():

    hts_dataset = HTSDataset(
        WORKDIR, transform=pad_data, max_size=DATASET_SIZE_LIMIT,
        f0_backward_window_len=F0_WINDOW_LEN, min_f0=MIN_F0, max_f0=MAX_F0,
        min_duration=MIN_DURATION, max_duration=MAX_DURATION)

    hprint('Dataset size: {}'.format(str(len(hts_dataset))))

    train_data = gluon.data.DataLoader(
        hts_dataset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT)
    net = build_net()
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    if TRAIN_ON_GPU:
        net.collect_params().reset_ctx(mx.gpu(0))
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})
    l2loss = mx.gluon.loss.L2Loss()

    hprint("Training... ({} epochs)".format(EPOCHS))

    loss_sequence = []
    for e in range(EPOCHS):
        cumulative_loss = 0
        for X_batch, y_batch in train_data:
            with autograd.record():
                output = net(X_batch)
                loss = l2loss(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)
            mean_loss = nd.mean(loss).asscalar()
            cumulative_loss += mean_loss

        hprint("Epoch {}, loss: {}".format(

            e, cumulative_loss / y_batch[0].shape[0]))
        loss_sequence.append(cumulative_loss / y_batch[0].shape[0])
    f = plt.figure()
    plt.plot(loss_sequence)
    f.savefig("plots/loss_sequence_{}.pdf".format(
        datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")),
        bbox_inches='tight')

    net.save_parameters('models/{}.params'.format(
        datetime.datetime.now().strftime("%B_%d_%Y_%I%M%p")))

    test(net, hts_dataset.test_data)


if __name__ == '__main__':
    train()
