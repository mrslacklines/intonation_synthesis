#!flask/bin/python

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import matplotlib.pyplot as plt

from tsviz import VisualizationToolbox
from utils_tsviz import *

try:
    import cPickle as pickle  # Python2
except ImportError:
    import pickle  # Python3

# Initialize the model
print("Initializing the model")
visualizationToolbox = VisualizationToolbox()
visualizationToolbox.initializeTensorflow()

# Spawn the service
print("Initializing the service")
app = Flask(__name__)
# cors = CORS(app)


@app.route('/viz/api/train', methods=['GET'])
def switchToTrain():
    if visualizationToolbox.switchToTrain():
        return jsonify({'status': 'ok'})


@app.route('/viz/api/test', methods=['GET'])
def switchToTest():
    if visualizationToolbox.switchToTest():
        return jsonify({'status': 'ok'})


@app.route('/viz/api/reset', methods=['GET'])
def resetIterator():
    if visualizationToolbox.resetIterator():
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'Unable to reset iterator'})


@app.route('/viz/api/set', methods=['GET'])
def setIterator():
    if 'iterator' in request.args:
        iterator = int(request.args.get('iterator', 0))
        success = visualizationToolbox.setIterator(iterator)
        output = {'status': 'ok'} if success else {'status': 'error', 'msg': 'Iterator exceeded the maximum possible value'}
        return jsonify(output)
    else:
        return jsonify({'status': 'error', 'msg': 'Iterator value not found'})


@app.route('/viz/api/prev', methods=['GET'])
def loadPreviousInput():
    if visualizationToolbox.prevInputButton():
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'No previous examples in the dataset'})


@app.route('/viz/api/next', methods=['GET'])
def loadNextInput():
    if visualizationToolbox.nextInputButton():
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'No more examples in the dataset'})


@app.route('/viz/api/get_iterator', methods=['GET'])
def getIterator():
    it = visualizationToolbox.getIterator()
    return jsonify({'status': 'ok', 'iterator': it})


@app.route('/viz/api/get_example', methods=['GET'])
def getExample():
    X, y = visualizationToolbox.getExample()
    return jsonify({'status': 'ok', 'x': X.T.tolist(), 'y': y.tolist()})


@app.route('/viz/api/get_prediction', methods=['GET'])
def getPrediction():
    X, y, pred, saliency = visualizationToolbox.getPrediction()
    return jsonify({'status': 'ok', 'x': X.T.tolist(), 'y': y.tolist(), 'pred': pred.tolist(), 'saliency': saliency.tolist()})


@app.route('/viz/api/get_architecture', methods=['GET'])
def getArchitecture():
    arch = visualizationToolbox.getArchitecture()
    return jsonify({'status': 'ok', 'architecture': arch})


@app.route('/viz/api/set_percentile', methods=['GET'])
def setPercentile():
    if 'percentile' in request.args:
        percentile = float(request.args.get('percentile', -1))
        visualizationToolbox.changePercentile(percentile)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'Percentile value not found'})


@app.route('/viz/api/fetch', methods=['GET'])
def fetchData():
    return jsonify({'data': visualizationToolbox.loadData()})


@app.route("/viz/api/classify", methods=['GET'])
def classify():
    dataShape = visualizationToolbox.trainX.shape[1:]
    seq = ";".join([",".join(['0' for i in range(dataShape[0])]) for j in range(dataShape[1])])

    if 'seq' in request.args:
        seq = request.args.get('seq', seq)

    seq = seq.split(";")
    seq = np.array([list(eval(x)) for x in seq]).T
    return jsonify({'status': 'ok', 'prediction': str(visualizationToolbox.classifySequence(seq))})


@app.route('/viz/api/cluster', methods=['GET'])
def performClustering():
    visualizationToolbox.performFilterClustering()
    return jsonify({'status': 'ok'})


@app.route('/viz/api/visualize_clusters', methods=['GET'])
def visualizeClusters():
    visualizationToolbox.visualizeFilterClusters()
    return jsonify({'status': 'ok'})


@app.route("/viz/api/test_model", methods=["GET"])
def testCurrentModel():
    result = visualizationToolbox.performTesting()
    return jsonify({'status': 'ok', 'Result': result})


@app.route("/viz/api/load_default_model", methods=['GET'])
def loadDefaultModel():
    visualizationToolbox.modelPath = visualizationToolbox.standardModelPath
    visualizationToolbox.currentlyLoadedModel = visualizationToolbox.standardModelName
    visualizationToolbox.loadModel()
    # TODO: Importance values for number of examples
    #visualizationToolbox.loadImportanceValues(visualizationToolbox.standardModelName)
    return jsonify({'status': 'ok'})


@app.route("/viz/api/get_loaded_model_name", methods=['GET'])
def getLoadedModelName():
    return jsonify({'status': 'ok', 'name': visualizationToolbox.currentlyLoadedModel})



@app.route("/viz/api/get_filter_list", methods=['GET'])
def getFilterList():
    #if datamode==1 ? compute for all examples : compute for selected
    dataMode = int(request.args.get("data_mode", Modes.SELECTED_EXAMPLES.value))

    # if importance_mode==0 ? percentile maximum
    # if importance_mode==1 ? percentile minimum
    # if importance_mode==2 ? sorted importance maximum
    # if importance_mode==3 ? sorted importance miminum
    # if importance_mode==4 ? cluster representatives
    importanceMode = int(request.args.get("importance_mode", Modes.PERCENTILE_MAXIMUM.value))

    # if importance_mode==(0 or 1) ? percentile value
    # if importance_mode==(2 or 3) ? number of top most results
    numberOfFilter = int(request.args.get("number_of_filter", 0))

    # list of layerids included in the process e.g. network =[conv,max,conv,max] -> 0,1 to process both conv layers
    # the number is not the actual layer, it is the number of the convlayer.
    # first conv layer in the network is 0 and second is 1 (does not depend on other layers between those two cons layers)
    layerSelection = np.fromstring(request.args.get("layer_selection", "0"), dtype=int, sep=",")

    # if importanceSelection==0 ? importance : loss
    importanceSelection = int(request.args.get("importanceSelection", Modes.IMPORTANCE.value))

    # if dataset==1 ? testset : trainset
    dataset = int(request.args.get("dataset", Dataset.TRAIN.value))

    result = visualizationToolbox.computePruningFilterSet(dataMode, importanceMode, numberOfFilter, layerSelection, importanceSelection, dataset)
    result = listToIndiceString(result)
    return jsonify({'status': 'ok', 'indices': result})


@app.route("/viz/api/get_filter_list_from_file", methods=['GET'])
def getFilterListFromFile():
    # if mode==0 ? compute random
    # if mode==1 ? compute percentile
    # if mode==2 ? compute representative
    mode = int(request.args.get("mode", Modes.COMPUTE_RANDOM.value))

    # if submode==0 ? min
    # if submode==1 ? max
    # if submode==2 ? mean
    submode = int(request.args.get("submode", Modes.MEAN.value))

    # if mode==(0 or 1) ? percentile
    percentile = int(request.args.get("percentile", 10))

    # if mode==(1 or 2) ? lowest==0 or most important==1
    reverse = int(request.args.get("reverse", 0))

    # if mode==(1 or 2) ? number of examples
    examples = int(request.args.get("examples", 100))

    # if importanceSelection==0 ? importance : loss
    importanceSelection = int(request.args.get("importanceSelection", Modes.IMPORTANCE.value))

    # if dataset==1 ? testset : trainset
    dataset = int(request.args.get("dataset", Dataset.TRAIN.value))

    result = visualizationToolbox.computePruningFilterSetFromFile(mode, submode, percentile, reverse, examples, importanceSelection, dataset)
    result = listToIndiceString(result)
    return jsonify({'status': 'ok', 'indices': result})


if __name__ == '__main__':
    # Create the required directories
    ensureDirExists("./prunedModels")
    ensureDirExists("./adjustedFilters")
    ensureDirExists("./Statistics")

    # Start the service
    print("Starting the service")
    app.run(host="0.0.0.0", port=5000)
