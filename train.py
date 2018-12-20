import mxnet as mx
import numpy
import os
from merlin.src.io_funcs import htk_io
from multiprocessing import cpu_count
from mxnet import nd, autograd, gluon
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from sklearn.preprocessing import MinMaxScaler, Normalizer


CPU_COUNT = cpu_count()
WORKDIR = '/media/tomaszk/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT'
DATA_SIZE = 1


def train():
    binary_dict, continuous_dict = hts.load_question_set(
        WORKDIR + '/data/questions/questions_qst001.hed')

    model_ctx = mx.cpu()

    X = numpy.ndarray([DATA_SIZE, 401, 1526])
    y = numpy.ndarray([DATA_SIZE, 401])

    for index, label_file in enumerate(
            os.listdir(WORKDIR + '/data/labels/full/')):
        basename, ext = os.path.splitext(label_file)

        if index == DATA_SIZE:
            break

        if 'lab' not in ext:
            continue

        full = hts.load(os.path.join(WORKDIR, 'data/labels/full/', label_file))

        linguistic_features = fe.linguistic_features(
            full, binary_dict, continuous_dict, subphone_features=None,
            add_frame_features=True)

        htk_reader = htk_io.HTK_Parm_IO()
        htk_reader.read_htk(WORKDIR + '/data/cmp/' + basename + '.cmp')

        target = htk_reader.data[:, 150].copy()
        target = numpy.nan_to_num(target, copy=True)
        # mark unvoiced regions with 0 instead of -inf
        numpy.place(target, target < 0, [None, ])
        acoustic_features = numpy.delete(htk_reader.data, 150, 1)

        acoustic_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            acoustic_features)
        acoustic_features = Normalizer().fit_transform(acoustic_features)

        features = numpy.hstack([acoustic_features, linguistic_features])

        assert(features.shape[0] == target.shape[0])

        X[index] = features
        y[index] = target

    batch_size = 1

    train_data = gluon.data.DataLoader(
        gluon.data.ArrayDataset(X, y),
        batch_size=batch_size, num_workers=CPU_COUNT)

    num_of_feats = features.shape[1]

    net = mx.gluon.nn.Sequential()
    with net.name_scope():
        net.add(mx.gluon.rnn.LSTM(
            hidden_size=100, bidirectional=True, layout='NTC'))
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': 0.0001})
    smloss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    epochs = 500
    loss_sequence = []
    num_batches = target.shape[0] / batch_size

    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            label = mx.nd.array(numpy.nan_to_num(label.asnumpy()))
            data = data.as_in_context(model_ctx).astype(numpy.float32)
            label = label.as_in_context(model_ctx).astype(numpy.float32)
            with autograd.record():
                output = net(data.astype(numpy.float32))
                loss = smloss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.mean(loss).asscalar()
        print("Epoch %s, loss: %s" % (e, cumulative_loss / target.shape[0]))
        loss_sequence.append(cumulative_loss)


if __name__ == '__main__':
    train()
