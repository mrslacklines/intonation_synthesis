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
EPOCHS = 100
LEARNING_RATE = 0.0001

NUM_LAYERS = 3
HIDDEN_SIZE = 100
DROPOUT = 0.2
BIDIRECTIONAL = True


model_ctx = mx.cpu()


def pad_data(features, target, max_seq_len=1900):
    X = numpy.ndarray([max_seq_len, 1524])
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

    train_data = gluon.data.DataLoader(
        HTSDataset(WORKDIR, transform=pad_data, max_size=DATASET_SIZE_LIMIT),
        batch_size=BATCH_SIZE, num_workers=CPU_COUNT)

    net = build_net()
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': LEARNING_RATE})
    sceloss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    print("=" * 50)
    print("Training... ({} epochs)".format(EPOCHS))
    print("=" * 50)

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
        print("Epoch {}, loss: {}".format(
            e, cumulative_loss / y_batch[0].shape[0]))
        loss_sequence.append(cumulative_loss / y_batch[0].shape[0])

    plt.plot(loss_sequence)
    plt.show()


if __name__ == '__main__':
    train()
