import mxnet as mx
import numpy
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from mxnet import nd, autograd, gluon

from datasets import HTSDataset
from utils import pad_array


CPU_COUNT = cpu_count()
WORKDIR = '/media/tomaszk/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT'
BATCH_SIZE = 10
DATASET_SIZE_LIMIT = None
EPOCHS = 200
LEARNING_RATE = 0.0001

NUM_LAYERS = 3
HIDDEN_SIZE = 100
DROPOUT = 0.2
USE_MOVING_WINDOW = True
F0_WINDOW_LEN = 20
BIDIRECTIONAL = True if not USE_MOVING_WINDOW else False
TRAIN_ON_GPU = False
FEATURES_ORDER = 4114

MAX_DURATION = 14950000.0
MIN_DURATION = 0
MIN_F0 = -20.299814419936542
MAX_F0 = 35.15619563784128


model_ctx = mx.cpu()


def hprint(string):
    print('=' * 50)
    print(string)
    print('=' * 50)



def pad_data(features, target, max_seq_len=1900):
    X = numpy.ndarray([max_seq_len, FEATURES_ORDER])
    y = numpy.ndarray([max_seq_len])
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

    return net


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
    sceloss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    hprint("Training... ({} epochs)".format(EPOCHS))

    loss_sequence = []
    for e in range(EPOCHS):
        cumulative_loss = 0
        for X_batch, y_batch in train_data:
            with autograd.record():
                output = net(X_batch)
                loss = sceloss(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)
            mean_loss = nd.mean(loss).asscalar()
            cumulative_loss += mean_loss

        hprint("Epoch {}, loss: {}".format(

            e, cumulative_loss / y_batch[0].shape[0]))
        loss_sequence.append(cumulative_loss / y_batch[0].shape[0])
        f = plt.figure()
        plt.plot(loss_sequence)
        f.savefig("loss_sequence.pdf", bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    train()
