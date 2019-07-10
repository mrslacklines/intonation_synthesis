import os
from multiprocessing import cpu_count


DEBUG = True

CPU_COUNT = round(cpu_count() / 2)
WORKDIR = 'data/' if DEBUG else '/opt/ml/'

DATASET_SIZE_LIMIT = None

TRAIN_ON_GPU = True if not DEBUG and os.environ['TRAIN_ON_GPU'] else False
MULTI_PRECISION = False
