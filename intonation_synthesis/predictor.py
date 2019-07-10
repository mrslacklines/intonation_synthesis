import flask
import json
import mxnet as mx
import os
import pandas as pd
from io import StringIO

import settings
from net import build_net


model_path = os.path.join(settings.WORKDIR, 'model')


with open(
        settings.WORKDIR + '/input/config/hyperparameters.json') as json_file:
    hyperparameters = json.load(json_file)

BATCH_SIZE = int(hyperparameters.get('batch_size'))
EPOCHS = int(hyperparameters.get('epochs'))

HIDDEN_SIZE = int(hyperparameters.get('hidden_size'))
NUM_LAYERS = int(hyperparameters.get('num_layers'))
DROPOUT = float(hyperparameters.get('dropout'))
LEARNING_RATE = float(hyperparameters.get('learning_rate'))
BIDIRECTIONAL = bool(hyperparameters.get('bidirectional'))


class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = build_net(
                HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, BATCH_SIZE,
                mx.cpu(), settings.MULTI_PRECISION)
        cls.model.load_parameters(
            settings.WORKDIR + '/model/intonation.params')
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    # You can insert a health check here
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(
        response='', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(
            response='This predictor only supports CSV data', status=415,
            mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    predictions = ScoringService.predict(data)

    out = StringIO.StringIO()
    pd.DataFrame({'results': predictions}).to_csv(
        out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)