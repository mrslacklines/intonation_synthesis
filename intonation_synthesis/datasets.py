import numpy
import os
import pandas
import random
import re
import tensorflow as tf
from merlin.io_funcs import htk_io
from merlin.io_funcs.binary_io import BinaryIOCollection
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts

import const
from utils import scale


class HTSDataset(tf.keras.utils.Sequence):
    """
    A dataset for loading HTS data from disk.
    """

    def __init__(
            self, hts_dataset_path, batch_size=1, file_list=None, transform=None,
            max_size=None, split_ratio=0.8, randomize=True,
            f0_backward_window_len=50, min_f0=None, max_f0=None,
            min_duration=None, max_duration=None, rich_feats=False):

        self.batch_size = batch_size
        self.step = self.batch_size if self.batch_size > 1 else 1
        self.max_size = max_size
        self.split_ratio = split_ratio
        self.randomize = randomize
        self.target_feature_order = 150
        self.target_features_list = [150, 151, 152]
        self.f0_backward_window_len = f0_backward_window_len
        self.workdir = hts_dataset_path
        self.file_list = file_list
        self._transform = transform
        self.question_file_name = \
            self.workdir + '/data/questions/questions_qst001.hed'
        # Smaller set of contextual linguistic features:
        # self.question_file_name = \
        #     self.workdir + '/data/questions/questions_qst001_no_phoneme_id.hed'
        self.list_files()
        self.load_hed_questions()

        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.PAU = re.compile('\-pau\+')
        self.LAST_SEG_IN_SYL = re.compile('_1\/A\:')
        self.LAST_SYL_IN_WORD = re.compile('\-1\&')
        self.LAST_SYL_IN_PHR = re.compile('\-1\#')
        self.SYL_POS_IN_PHR = re.compile('\-(\d)\#')

        self.rich_feats = rich_feats

        self.current_index = 0

    def list_files(self):
        if self.file_list is not None:
            self.data = self.file_list
        elif self.file_list is None:
            file_list = [
                file for file in os.listdir(
                    self.workdir + '/data/labels/full/')
                if 'lab' in os.path.splitext(file)[1]]
            if self.randomize:
                random.shuffle(file_list)
            if self.max_size:
                file_list = file_list[:self.max_size]

            self.data = file_list[:int(len(file_list) * self.split_ratio)]
            self._holdout_data = \
                file_list[int(len(file_list) * self.split_ratio):]
            self.validation_data = \
                self._holdout_data[:int(len(self._holdout_data) / 2)]
            self.test_data = \
                self._holdout_data[int(len(self._holdout_data) / 2):]

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

        return features.astype(numpy.bool)

    def add_f0_window(self, f0):
        df = pandas.DataFrame(f0, columns=['f0', ])
        for val_num in range(1, self.f0_backward_window_len + 1):
            for i in range(val_num, len(df)):
                df.loc[i, 'f0_lookback' + str(val_num)] = \
                    df.loc[i - val_num, 'f0']
        df.fillna(0, inplace=True)
        df = df.drop('f0', axis=1)
        if self.min_f0 is not None and self.max_f0 is not None:
            df = df.apply(scale, old_min=self.min_f0, old_max=self.max_f0)
        return df.values

    def make_vuv(self, f0):
        return numpy.array(
            [[el, ] for el in numpy.isfinite(f0).astype(int)]).astype(
            numpy.bool)

    def load_lf0(self, filename):
        binary_reader = BinaryIOCollection()
        return binary_reader.load_binary_file(
            self.workdir + '/data/lf0/' + filename,
            1).flatten()

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

        ref_df = df.copy()
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
        for index, row in ref_df.iterrows():
            for column in ref_df.columns:
                if row[column] != 0:
                    rem_duration = row[column]
                    for prev_index in range(
                            row.name - row[column] + frame_shift_in_micro_sec,
                            row.name + frame_shift_in_micro_sec,
                            frame_shift_in_micro_sec):
                        df.loc[prev_index, column + '_remaining'] = \
                            rem_duration
                        rem_duration -= frame_shift_in_micro_sec

        df.fillna(0, inplace=True)

        if self.min_duration is not None and self.max_duration is not None:
            df = df.apply(
                scale, old_min=self.min_duration, old_max=self.max_duration)

        return df.values

    def add_ti_features(self, f0):
        df = pandas.DataFrame(f0, columns=['f0_' + str(n) for n in range(
            self.f0_backward_window_len)])
        df.fillna(0, inplace=True)
        if self.min_f0 is not None and self.max_f0 is not None:
            df = df.apply(scale, old_min=self.min_f0, old_max=self.max_f0)
        if self.rich_feats:
            # inline import only in case we want to use TA
            # prevents massive installs on AWS containers
            try:
                from technical_indicators import make_technical_indicators
            except ImportError:
                raise ImportError(
                    'Please make sure TALib is installed properly..')
        df = make_technical_indicators(df)
        return df

    def make_data_for_index(self, idx):
        filename = self.data[idx]
        # DEBUG:
        # print("Processing {}..".format(filename))
        basename, ext = os.path.splitext(filename)
        fullcontext_label = self.load_hts_label(filename)
        linguistic_features = self.load_linguistic_features(fullcontext_label)
        acoustic_features, target = self.load_hts_acoustic_data(
            basename + '.cmp')

        if target.shape[0] == (linguistic_features.shape[0] + 1):
            target = target[:-1]
        vuv = self.make_vuv(target)
        target = target.reshape(-1, 1)

        assert(
            target.shape[0] ==
            linguistic_features.shape[0] ==
            vuv.shape[0])
        feats = numpy.hstack(
            [linguistic_features, vuv])

        if self._transform is not None:
            feats, target = self._transform(feats, target)

        return \
            numpy.nan_to_num(feats).astype(numpy.bool), \
            numpy.nan_to_num(target).astype(numpy.float32)

    def __getitem__(self, idx):

        X_list = []
        y_list = []

        for item in range(self.batch_size):
            file_idx = (idx * self.batch_size) + item
            X, y = self.make_data_for_index(file_idx)
            X_list.append(X)
            y_list.append(y)

        self.current_index += self.step

        return numpy.stack(X_list), numpy.stack(y_list)

    def __len__(self):
        return len(self.data) \
            if self.batch_size <= 0 else int(len(self.data) / self.batch_size)
