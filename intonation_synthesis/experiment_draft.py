import json

from mxnet import gluon
from tensorflow import keras

import const, settings
from datasets import HTSDataset
from train import BATCH_SIZE, CPU_COUNT, pad_data


model = keras.models.load_model('/opt/ml/model/model.model')
with open('/opt/ml/model/training_report.txt', 'r+') as json_report:
    report = json.load(json_report)
hts_validation_dataset = HTSDataset(
    settings.WORKDIR + 'input/data/training', file_list=report['holdout_data'],
    transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
    min_f0=const.MIN_F0, max_f0=const.MAX_F0,
    min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
    rich_feats=const.RICH_FEATS)
validation_data = gluon.data.DataLoader(
    hts_validation_dataset, batch_size=BATCH_SIZE, num_workers=CPU_COUNT,
    last_batch='discard')

for X_batch, y_batch in validation_data:
    predictions = model.predict(X_batch.asnumpy())
    import ipdb; ipdb.set_trace()  # breakpoint 329ccd61 //
    pass