import mxnet as mx
import numpy
import os
import pandas
import random
import re
from merlin.io_funcs import htk_io
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from sklearn.preprocessing import MinMaxScaler, Normalizer

from utils import interpolate_f0


class HTSDataset(mx.gluon.data.dataset.Dataset):
    """
    A dataset for loading HTS data from disk.
    """

    def __init__(
            self, hts_dataset_path, transform=None, max_size=None,
            split_ratio=0.8, randomize=True):
        self.max_size = max_size
        self.split_ratio = split_ratio
        self.randomize = randomize
        self.target_feature_order = 150
        self.target_features_list = [150, 151, 152]
        self.workdir = hts_dataset_path
        self._transform = transform
        self.question_file_name = \
            self.workdir + '/data/questions/questions_qst001.hed'
        self.list_files()
        self.load_hed_questions()

        self.PAU = re.compile('\-pau\+')
        self.LAST_SEG_IN_SYL = re.compile('_1\/A\:')
        self.LAST_SYL_IN_WORD = re.compile('\-1\&')
        self.LAST_SYL_IN_PHR = re.compile('\-1\#')
        self.SYL_POS_IN_PHR = re.compile('\-(\d)\#')

    def list_files(self):
        file_list = [
            file for file in os.listdir(self.workdir + '/data/labels/full/')
            if 'lab' in os.path.splitext(file)[1]]
        if self.randomize:
            random.shuffle(file_list)
        if self.max_size:
            file_list = file_list[:self.max_size]

        self.train_data = file_list[:int(len(file_list) * self.split_ratio)]
        self.test_data = file_list[int(len(file_list) * self.split_ratio):]

    def load_hed_questions(self):
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            self.question_file_name)

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

    def make_ti(self, f0):
        pass

    def load_hts_acoustic_data(
            self, cmp_filename, add_ti=False, use_window=False):
        htk_reader = htk_io.HTK_Parm_IO()
        htk_reader.read_htk(self.workdir + '/data/cmp/' + cmp_filename)

        target = htk_reader.data[:, self.target_feature_order].copy()
        target = numpy.nan_to_num(target, copy=True)
        # mark unvoiced regions with 0 instead of -inf
        numpy.place(target, target < 0, [None, ])
        acoustic_features = numpy.delete(
            htk_reader.data, self.target_features_list, 1)
        if add_ti:
            self.make_ti(interpolate_f0(target))
        return acoustic_features, target

    def load_duration_data(self, label, frame_shift_in_micro_sec=50000):

        labs = self.load_hts_label(label)

        seg_dur = 0
        syl_dur = 0
        word_dur = 0
        seg_dur_list = []
        syl_dur_list = []
        word_dur_list = []
        for index, (start, end, lab) in enumerate(labs):
            if self.PAU.search(lab):
                seg_dur_list.append(0)
                syl_dur_list.append(0)
                word_dur_list.append(0)
                seg_dur = 0
                syl_dur = 0
                word_dur = 0
                continue
            seg_dur = end - start
            syl_dur += seg_dur
            seg_dur_list.append(seg_dur)

            last_seg_in_syl = self.LAST_SEG_IN_SYL.search(lab)
            last_syl_in_word = self.LAST_SYL_IN_WORD.search(lab)
            cur_seg_syl_pos = \
                self.SYL_POS_IN_PHR.search(lab).groups()[0] if \
                self.SYL_POS_IN_PHR.search(lab) \
                and index <= len(labs) else -1
            next_seg_syl_pos = \
                self.SYL_POS_IN_PHR.search(labs[index + 1][2]).groups()[0] if \
                self.SYL_POS_IN_PHR.search(labs[index + 1][2]) \
                and index <= len(labs) else -1

            if last_seg_in_syl and \
                    int(cur_seg_syl_pos) != int(next_seg_syl_pos) or \
                    int(cur_seg_syl_pos) < 0:
                syl_dur_list.append(syl_dur)
                word_dur += syl_dur
                syl_dur = 0
                if last_syl_in_word:
                    word_dur_list.append(word_dur)
                    word_dur = 0
                else:
                    word_dur_list.append(0)
            else:
                syl_dur_list.append(0)
                word_dur_list.append(0)

        df = pandas.DataFrame(
            columns=['segment', 'syllable', 'word'],
            index=numpy.arange(
                start=frame_shift_in_micro_sec,
                stop=labs.num_frames() * frame_shift_in_micro_sec,
                step=frame_shift_in_micro_sec))
        df.fillna(value=0, inplace=True)

        for (start_time, end_time, lab), seg_dur, syl_dur, word_dur in zip(
                labs, seg_dur_list, syl_dur_list, word_dur_list):
            current_block_t = [seg_dur, syl_dur, word_dur]
            df.loc[end_time] = current_block_t

        last_seg_dur = 0
        last_syl_dur = 0
        last_word_dur = 0

        for index, row in df[::-1].iterrows():
            if row['segment'] > 0:
                last_seg_dur = row['segment']
            elif last_seg_dur:
                last_seg_dur -= frame_shift_in_micro_sec
                df.loc[index]['segment'] = last_seg_dur
            if row['syllable'] > 0:
                last_syl_dur = row['syllable']
            elif last_syl_dur:
                last_syl_dur -= frame_shift_in_micro_sec
                df.loc[index]['syllable'] = last_syl_dur

            if row['word'] > 0:
                last_word_dur = row['word']
            elif last_word_dur:
                last_word_dur -= frame_shift_in_micro_sec
                df.loc[index]['word'] = last_word_dur

        return df.values

    def preprocess_acoustic_features(self, acoustic_features):
        acoustic_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            acoustic_features)
        acoustic_features = Normalizer().fit_transform(acoustic_features)

        return acoustic_features

    def __getitem__(self, idx):
        filename = self.train_data[idx]
        basename, ext = os.path.splitext(self.train_data[idx])
        fullcontext_label = self.load_hts_label(filename)
        linguistic_features = self.load_linguistic_features(fullcontext_label)
        acoustic_features, target = self.load_hts_acoustic_data(
            basename + '.cmp')
        acoustic_features = self.preprocess_acoustic_features(
            acoustic_features)
        duration_features = self.load_duration_data(filename)
        assert(
            acoustic_features.shape[0] == duration_features.shape[0] ==
            linguistic_features.shape[0])
        feats = numpy.hstack(
            [acoustic_features, duration_features, linguistic_features])

        if self._transform is not None:
            feats, target = self._transform(feats, target)

        feats, target = \
            mx.nd.array(numpy.nan_to_num(feats)), \
            mx.nd.array(numpy.nan_to_num(target))
        return feats, target

    def __len__(self):
        return len(self.train_data)
