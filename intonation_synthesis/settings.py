import os
from multiprocessing import cpu_count


DEBUG = True

CPU_COUNT = round(cpu_count() / 2)
WORKDIR = '/opt/ml/'

DATASET_SIZE_LIMIT = None

try:
    TRAIN_ON_GPU = os.environ['TRAIN_ON_GPU'] and not DEBUG
except KeyError:
    TRAIN_ON_GPU = False

MULTI_PRECISION = False
