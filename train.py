import mxnet as mx
import numpy
import os
from matplotlib import pyplot as plt
from merlin.src.io_funcs import htk_io
from multiprocessing import cpu_count
from mxnet import nd, autograd, gluon
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from sklearn.preprocessing import MinMaxScaler, Normalizer

from utils import pad_array


CPU_COUNT = cpu_count()
WORKDIR = '/media/tomaszk/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT'
DATA_SIZE = 200
MAX_SEQUENCE_LEN = 1900
BATCH_SIZE = 5
EPOCHS = 100


model_ctx = mx.cpu()


def load_hed_questions():
    binary_dict, continuous_dict = hts.load_question_set(
        WORKDIR + '/data/questions/questions_qst001.hed')

    return binary_dict, continuous_dict


def load_linguistic_features(
        fullcontext_label, subphone_features=None, add_frame_features=True):
    binary_dict, continuous_dict = load_hed_questions()
    features = fe.linguistic_features(
        fullcontext_label, binary_dict, continuous_dict,
        subphone_features=subphone_features,
        add_frame_features=add_frame_features)

    return features


def load_hts_label(filename):
    lab = hts.load(os.path.join(WORKDIR, 'data/labels/full/', filename))

    return lab


def load_hts_acoustic_data(cmp_filename):
    htk_reader = htk_io.HTK_Parm_IO()
    htk_reader.read_htk(WORKDIR + '/data/cmp/' + cmp_filename)

    target = htk_reader.data[:, 150].copy()
    target = numpy.nan_to_num(target, copy=True)
    # mark unvoiced regions with 0 instead of -inf
    numpy.place(target, target < 0, [None, ])
    acoustic_features = numpy.delete(htk_reader.data, [150, 151, 152], 1)

    return acoustic_features, target


def preprocess_acoustic_features(acoustic_features):
    acoustic_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        acoustic_features)
    acoustic_features = Normalizer().fit_transform(acoustic_features)

    return acoustic_features


def train():

    X = numpy.ndarray([DATA_SIZE, MAX_SEQUENCE_LEN, 1526])
    y = numpy.ndarray([DATA_SIZE, MAX_SEQUENCE_LEN])

    print("=" * 50)
    print("Preparing data...")
    print("=" * 50)

    for index, label_file in enumerate(
            os.listdir(WORKDIR + '/data/labels/full/')):
        basename, ext = os.path.splitext(label_file)

        if index == DATA_SIZE:
            break

        if 'lab' not in ext:
            continue

        print('Processing {}...'.format(basename))

        fullcontext_label = load_hts_label(label_file)
        binary_dict, continuous_dict = load_hed_questions()
        linguistic_features = load_linguistic_features(fullcontext_label)
        acoustic_features, target = load_hts_acoustic_data(basename + '.cmp')
        acoustic_features = preprocess_acoustic_features(acoustic_features)
        features = numpy.hstack([acoustic_features, linguistic_features])

        assert(features.shape[0] == target.shape[0])

        features_padded = pad_array(X.shape[1:], features)
        target_padded = pad_array(y.shape[1], target)

        X[index] = features_padded
        y[index] = target_padded

    train_data = gluon.data.DataLoader(
        gluon.data.ArrayDataset(X, y),
        batch_size=BATCH_SIZE, num_workers=CPU_COUNT)

    net = mx.gluon.nn.Sequential()
    with net.name_scope():
        net.add(mx.gluon.rnn.LSTM(
            hidden_size=100, bidirectional=True, layout='NTC'))
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': 0.0001})
    sceloss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    print("=" * 50)
    print("Training... ({} epochs)".format(EPOCHS))
    print("=" * 50)

    loss_sequence = []
    for e in range(EPOCHS):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            label = mx.nd.array(numpy.nan_to_num(label.asnumpy()))
            data = data.as_in_context(model_ctx).astype(numpy.float32)
            label = label.as_in_context(model_ctx).astype(numpy.float32)
            with autograd.record():
                output = net(data.astype(numpy.float32))
                loss = sceloss(output, label)
            loss.backward()
            trainer.step(BATCH_SIZE)
            mean_loss = nd.mean(loss).asscalar()
            cumulative_loss += mean_loss
        print("Epoch %s, loss: %s" % (e, cumulative_loss / target.shape[0]))
        loss_sequence.append(cumulative_loss / target.shape[0])

    plt.plot(loss_sequence)
    plt.show()


if __name__ == '__main__':
    train()
