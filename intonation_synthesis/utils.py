import numpy
import os
import pandas

from datetime import datetime

from nnmnkwii.io import hts


def hprint(string):
    print('=' * 50)
    print(datetime.now())
    print(string)
    print('=' * 50)


def add_time_to_full_context_labels_from_fal(workdir):
    """
    Adding start and end times to HTS full context labels from fal labels
    """
    for label_file in os.listdir(workdir + '/data/labels/full/'):
        full = hts.load(os.path.join(workdir, 'data/labels/full/', label_file))
        fal = hts.load(os.path.join(workdir, 'gv/qst001/ver1/fal', label_file))
        if not os.path.exists(os.path.join(workdir, 'backup')):
            os.mkdir(os.path.join(workdir, 'backup'))
        assert len(fal) == len(full)
        full.start_times = fal.start_times
        full.end_times = fal.end_times
        with open(workdir + '/data/labels/full/' + label_file, 'w') as lab:
            lab.write(str(full))


def pad_array(ref_shape, array, pad_with=0):
    if pad_with == 1:
        result = numpy.ones(ref_shape)
    elif pad_with == 0:
        result = numpy.zeros(ref_shape)
    if len(array.shape) == 1:
        result[:array.shape[0]] = array
    elif len(array.shape) == 2:
        result[:array.shape[0], :array.shape[1]] = array
    return result


def interpolate_f0(f0_arr):
    series = pandas.Series(f0_arr, dtype='d')
    return series.interpolate(
        method='cubic').fillna(0).values


def scale(value, old_min, old_max, new_min=0, new_max=1):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    return (((value - old_min) * new_range) / old_range) + new_min
