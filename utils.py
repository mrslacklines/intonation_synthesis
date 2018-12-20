import os

from nnmnkwii.io import hts


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
