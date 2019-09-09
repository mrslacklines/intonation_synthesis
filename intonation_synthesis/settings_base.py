import mxnet as mx
import os
from multiprocessing import cpu_count


DEBUG = False

CPU_COUNT = 0  # round(cpu_count() / 2)
GPU_COUNT = mx.context.num_gpus()
WORKDIR = '/opt/ml/'

DATASET_SIZE_LIMIT = None

try:
    TRAIN_ON_GPU = os.environ['TRAIN_ON_GPU'] and not DEBUG
except Exception:
    TRAIN_ON_GPU = False

if TRAIN_ON_GPU:
    if GPU_COUNT > 1:
        MODEL_CTX = [mx.gpu(i) for i in range(GPU_COUNT)]
    else:
        MODEL_CTX = mx.gpu()
else:
    MODEL_CTX = mx.cpu()
