import mxnet as mx
import os
import numpy
from merlin.src.io_funcs import htk_io
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from sklearn.preprocessing import MinMaxScaler, Normalizer


mx.random.seed(42)  # set seed for repeatability


class HTSDataset(mx.gluon.data.dataset.Dataset):
    """
    A dataset for loading HTS data from disk.
    """

    def __init__(self, hts_dataset_path, transform=None, max_size=None):
        self.max_size = max_size
        self.target_feature_order = 150
        self.target_features_list = [150, 151, 152]
        self.workdir = hts_dataset_path
        self._transform = transform
        self.list_files()
        self.load_hed_questions()

    def list_files(self):
        file_list = [
            file for file in os.listdir(self.workdir + '/data/labels/full/')
            if 'lab' in os.path.splitext(file)[1]]
        if self.max_size:
            file_list = file_list[:self.max_size]

        self.file_list = file_list

    def load_hed_questions(self):
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            self.workdir + '/data/questions/questions_qst001.hed')

    def load_hts_label(self, filename):
        lab = hts.load(
            os.path.join(self.workdir, 'data/labels/full/', filename))

        return lab

    def load_linguistic_features(
            self, fullcontext_label, subphone_features=None,
            add_frame_features=True):
        features = fe.linguistic_features(
            fullcontext_label, self.binary_dict, self.continuous_dict,
            subphone_features=subphone_features,
            add_frame_features=add_frame_features)

        return features

    def load_hts_acoustic_data(self, cmp_filename):
        htk_reader = htk_io.HTK_Parm_IO()
        htk_reader.read_htk(self.workdir + '/data/cmp/' + cmp_filename)

        target = htk_reader.data[:, self.target_feature_order].copy()
        target = numpy.nan_to_num(target, copy=True)
        # mark unvoiced regions with 0 instead of -inf
        numpy.place(target, target < 0, [None, ])
        acoustic_features = numpy.delete(
            htk_reader.data, self.target_features_list, 1)

        return acoustic_features, target

    def preprocess_acoustic_features(self, acoustic_features):
        acoustic_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            acoustic_features)
        acoustic_features = Normalizer().fit_transform(acoustic_features)

        return acoustic_features

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        basename, ext = os.path.splitext(self.file_list[idx])
        fullcontext_label = self.load_hts_label(filename)
        linguistic_features = self.load_linguistic_features(fullcontext_label)
        acoustic_features, target = self.load_hts_acoustic_data(
            basename + '.cmp')
        acoustic_features = self.preprocess_acoustic_features(
            acoustic_features)
        feats = numpy.hstack([acoustic_features, linguistic_features])

        if self._transform is not None:
            feats, target = self._transform(feats, target)

        feats, target = \
            mx.nd.array(numpy.nan_to_num(feats)), \
            mx.nd.array(numpy.nan_to_num(target))
        return feats, target

    def __len__(self):
        return len(self.file_list)
