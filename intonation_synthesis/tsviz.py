#!/bin/python

import os

# Packages for DL
import tensorflow as tf
import h5py
import keras.backend as K
from keras.models import load_model, save_model
from keras.backend.tensorflow_backend import set_session
import pandas as pd
import numpy as np
import scipy.stats
import math
from tqdm import tqdm

from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Flatten, Dense, Lambda, Conv2DTranspose, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("white")
from utils_tsviz import *
from config_tsviz import *

import const
import settings
from datasets import HTSDataset
from train import DROPOUT, HIDDEN_SIZE, NUM_LAYERS, pad_data
from utils import get_tcn_model_layers

try:
    import cPickle as pickle  # Python2
except ImportError:
    import pickle  # Python3

# Ignore keras 2 API warnings
import warnings
warnings.filterwarnings("ignore")

# # Create a new session for keras
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))


TRAIN_DATASET_LEN = 200


def removeBatchDim(input):
    # Since we are computing gradient wrt to only tensor, the returned list should be of length 1
    assert(len(input) == 1)
    input = input[0]  # tf.Gradients returns a list
    assert(input.shape[0] == 1)
    return input[0]  # Batch size is always 1



def cummulateGradients(input):
    # Combine the gradient for each filter to determine the impact of each filter on the final outcome
    input = np.abs(input)  # Absolute values should have been taken
    if len(input.shape) > 1:  # No reduction for dense layers
        input = np.sum(input, axis=0)
    else:
        input = input.copy()
    return input


class VisualizationToolbox:
    def __init__(self):
        # Define the API data indices
        self.datasetNameIdx = 'datasetName'
        self.datasetTypeIdx = 'datasetType'
        self.classNamesIdx = 'classNames'
        self.inputFeaturesNamesIdx = 'inputFeaturesNames'
        self.layerNamesIdx = 'layerNames'
        self.inverseOptimizationIdx = 'inverseOptimization'
        self.adversarialExamplesIdx = 'adversarialExamples'
        self.inputLayerIdx = 'inputLayer'
        self.groundTruthIdx = 'groundTruth'

        self.rawValueIdx = 'rawValue'
        self.saliencyMapIdx = 'saliencyMap'
        self.filterImportanceIdx = 'filterImportance'
        self.filterSaliencyIdx = 'filterSaliency'
        self.filterMaskIdx = 'filterMask'
        self.filterLossImpactIdx = 'filterLossImpact'
        self.filterClusterIdx = 'filterCluster'

        # Define the class specific variables
        self.inputIterator = 0
        self.currentPercentileValue = 0.0

        self.setType = Dataset.TEST.value

        # Load and setup the model
        self.loadDatasets()
        self.loadModel()

        self.importanceValues = None

    def initializeTensorflow(self):
        # Create the TensorFlow session
        config = tf.ConfigProto(device_count = {'GPU': 0})
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        set_session(self.sess)

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

    def loadDatasets(self):
        self.modelPath = "/app/model/model.hdf5"
        self.standardModelPath = self.modelPath
        self.standardModelName = "model"
        num_features = 1297 + 1
        with open('/app/model/training_report.txt', 'r') as report_file:
            validation_data = \
                [line.strip("\n") for line in report_file.readlines()]

        train_data = [
            lab for lab in os.listdir(
                "/opt/ml/input/data/training/data/labels/full/")
            if lab.endswith(".lab") and lab not in validation_data
        ]
        hts_train_dataset = HTSDataset(
            settings.WORKDIR + 'input/data/training',
            file_list=train_data,
            transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
            min_f0=const.MIN_F0, max_f0=const.MAX_F0,
            min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
            rich_feats=const.RICH_FEATS)
        self.hts_validation_dataset = HTSDataset(
            settings.WORKDIR + 'input/data/training',
            file_list=validation_data,
            transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
            min_f0=const.MIN_F0, max_f0=const.MAX_F0,
            min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
            rich_feats=const.RICH_FEATS)
        self.trainX, self.trainY = [], []
        self.testX, self.testY = [], []
        self.valX, self.valY = [], []
        self.inputFeaturesNames = [
            " ".join([exp.pattern for exp in value])
            for key, value in self.hts_validation_dataset.binary_dict.items()]

        for index in range(TRAIN_DATASET_LEN):
            X, y = hts_train_dataset[index]
            self.trainX.append(X.astype("float32"))
            self.trainY.append(y.astype("float32"))
        for index in range(int(len(self.hts_validation_dataset) / 2)):
            X, y = self.hts_validation_dataset[index]
            self.testX.append(X.astype("float32"))
            self.testY.append(y.astype("float32"))
        for index in range(int(len(self.hts_validation_dataset) / 2), len(self.hts_validation_dataset)):
            X, y = self.hts_validation_dataset[index]
            self.valX.append(X.astype("float32"))
            self.valY.append(y.astype("float32"))

        self.trainX = np.concatenate(self.trainX, axis=0)
        self.trainY = np.concatenate(self.trainY, axis=0)
        self.testX = np.concatenate(self.testX, axis=0)
        self.testY = np.concatenate(self.testY, axis=0)
        self.valX = np.concatenate(self.valX, axis=0)
        self.valY = np.concatenate(self.valY, axis=0)

    def loadModel(self):
        from net import build_net
        self.model = build_net(HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
        self.model.load_weights(self.modelPath)
        self.currentlyLoadedModel = self.standardModelPath

        print("Model summary:")
        self.model.summary()

        # Get the input placeholder
        self.inputPlaceholder = self.model.input  # Only single input
        outputLayers = get_tcn_model_layers(self.model)
        requiredGradients = [currentLayer.output[0] if isinstance(currentLayer.output, list) else currentLayer.output for currentLayer in outputLayers]
        self.outputLayers = outputLayers
        self.requiredGradients = requiredGradients
        self.outputLayer = requiredGradients[-1]


        self.labelsPlaceholder = tf.placeholder(tf.float32, shape=[None, 1])
        self.loss = tf.reduce_mean(tf.square(self.labelsPlaceholder - requiredGradients[-1])) # Regression loss

        self.layerType = []
        self.defaultLayerNames = []

        self.layerGradientsWrtInput = []
        self.layerNames = []
        self.numLayerFilters = []
        self.slicedTensors = []
        for layerIdx, tensorToBeDifferentiated in enumerate(requiredGradients):
            if isinstance(tensorToBeDifferentiated, list):
                residualModelTensor, skipConnectionTensor = tensorToBeDifferentiated
                tensorToBeDifferentiated = residualModelTensor
            layerShape = tensorToBeDifferentiated.shape
            layerName = getShortName(outputLayers[layerIdx].name, outputLayers[layerIdx].__class__.__name__)
            self.defaultLayerNames.append(layerName)
            self.layerType.append(outputLayers[layerIdx].__class__.__name__)
            print("Layer name: %s | Shape: %s | Type: %s" % (self.defaultLayerNames[-1], str(layerShape), self.layerType[-1]))
            self.numLayerFilters.append(int(layerShape[-1]))
            for filterIdx in range(layerShape[-1]):
                if len(layerShape) == 3:
                    tensorSlice = tf.expand_dims(tensorToBeDifferentiated[:, :, filterIdx], axis=-1)
                elif len(layerShape) == 2:
                    tensorSlice = tf.expand_dims(tensorToBeDifferentiated[:, filterIdx], axis=-1)
                else:
                    print("Error: Unknown tensor shape")
                    exit(-1)
                self.slicedTensors.append(tensorSlice)
                self.layerGradientsWrtInput.append(tf.gradients(tensorSlice, [self.inputPlaceholder]))
                self.layerNames.append(outputLayers[layerIdx].name + "-" + str(filterIdx))

        self.outputGradientsWrtLayer = [tf.gradients(self.outputLayer, [tensorToBeDifferentiated]) for tensorToBeDifferentiated in requiredGradients[:-1]]
        self.lossGradientsWrtLayer = [tf.gradients(self.loss, [tensorToBeDifferentiated]) for tensorToBeDifferentiated in requiredGradients[:-1]]

        # For inverse optimization and adversarial examples
        self.lossGrad = tf.gradients(self.loss, [self.inputPlaceholder])
        self.signGrad = tf.sign(self.lossGrad) # Use only the sign of the gradient
        self.outputGrad = tf.gradients(self.outputLayer, [self.inputPlaceholder])

        self.scaleGradientWrtLossValues = True


    def resetIterator(self):
        self.inputIterator = 0
        return True


    def setIterator(self, iterator):
        if (self.setType == Dataset.TEST.value and (self.inputIterator >= self.testX.shape[0] - 1)) or \
                (self.setType == Dataset.TRAIN.value and (self.inputIterator >= self.trainX.shape[0] - 1)):
            return False

        self.inputIterator = iterator
        return True


    def getIterator(self):
        return self.inputIterator


    def getModelLayerNames(self):
        if "input" not in self.layerNames[0]:
            return self.defaultLayerNames
        else:
            return self.defaultLayerNames[1:]


    def getExample(self):
        if self.setType == Dataset.TEST.value:
            X = self.testX[self.inputIterator, :, :]
            y = self.testY[self.inputIterator]
        else:
            X = self.trainX[self.inputIterator, :, :]
            y = self.trainY[self.inputIterator]
        return X, y


    def switchToTest(self):
        self.inputIterator = 0
        self.setType = Dataset.TEST.value
        return True


    def switchToTrain(self):
        self.inputIterator = 0
        self.setType = Dataset.TRAIN.value
        return True


    def getPrediction(self):
        # Get the session from Keras backend
        self.sess = K.get_session()

        X, y = self.getExample()

        # Get the prediction and the saliency for the last layer
        pred, saliency = self.sess.run([self.outputLayer, self.layerGradientsWrtInput[-1]], feed_dict={self.inputPlaceholder: np.expand_dims(X, axis=0)})
        pred = np.squeeze(pred)
        saliency = np.squeeze(saliency).T
        saliency = np.abs(saliency)  # For absolute scaling
        saliency = normalizeValues(saliency)

        return X, y, pred, saliency


    def getArchitecture(self):
        arch = []

        # Layer name, type, # filters, filter size, filter stride, output shape
        for layer in self.model.layers:
            arch.append(layer.get_config())

        return arch


    def loadData(self, fastMode=Modes.FULL.value, visualizationFilters=False, verbose=False):
        # Get the session from Keras backend
        self.sess = K.get_session()

        serviceOutput = h5py.File("/opt/ml/model/output.hdf5", "w")
        if self.setType == Dataset.TEST.value:
            currentInput = self.testX[self.inputIterator, :, :]
            currentLabel = self.testY[self.inputIterator]
        else:
            currentInput = self.trainX[self.inputIterator, :, :]
            currentLabel = self.trainY[self.inputIterator]

        # Append dataset type
        serviceOutput.attrs['datasetName'] = str(DATASET.name)
        serviceOutput.attrs['datasetType'] = str(DATASET.name)
        serviceOutput.attrs['classNames'] = CLASS_NAMES
        serviceOutput.attrs['inputFeaturesNames'] = self.inputFeaturesNames

        # layerNames[0] contains "input" if a pruned model has been loaded
        model_layer_names = np.string_(self.getModelLayerNames())
        serviceOutput.create_dataset('layerNames', data=model_layer_names)

        # Add inverse optimization and adversarial examples output here
        # startingSeries, startingSeriesForecast, startSerSaliencyMap, invOptimizedSeries, invOptimizedForecast, invOptSaliencyMap, advExampleOrig, forecastValueAdvOrig, advExSaliencyMap = self.performInverseOptimizationAndAdvAttack()

        # serviceOutput.create_dataset(
        #     'inverseOptimization/startingSeries', data=startingSeries)

        # serviceOutput.create_dataset(
        #     'inverseOptimization/startingSeriesForecast',
        #     data=startingSeriesForecast)

        # serviceOutput.create_dataset(
        #     'inverseOptimization/startSerSaliencyMap',
        #     data=startSerSaliencyMap)

        # serviceOutput.create_dataset(
        #     'inverseOptimization/invOptimizedSeries', data=invOptimizedSeries)

        # serviceOutput.create_dataset(
        #     'inverseOptimization/invOptimizedForecast',
        #     data=invOptimizedForecast)

        # serviceOutput.create_dataset(
        #     'inverseOptimization/invOptSaliencyMap',
        #     data=invOptSaliencyMap)

        # serviceOutput.create_dataset(
        #     'adversarialExamples', data="Not computed")

        # Add the raw input and the label
        serviceOutput.create_dataset('inputLayer', data=currentInput.T)

        serviceOutput.create_dataset('groundTruth', data=currentLabel.T)

        serviceOutput.create_group("data")
        # Iterate over the layers
        layerIdx = 0
        plotIdxRow = 0
        plotIdxCol = 0
        prevLayerName = None
        for idx, inputGradients in enumerate(self.layerGradientsWrtInput):
            currentLayerRootName = self.layerNames[idx]
            currentLayerRootName = currentLayerRootName[:currentLayerRootName.rfind('-')]

            computeGradientWrtInputOnly = True

            # If new layer encountered
            if (prevLayerName != currentLayerRootName):
                # # Add the min and max bounds for the plot (vertical)
                # self.plotBounds.append((self.currentPosition[2], self.currentPosition[2] + self.plotHeight))

                currentGroupName = "data/{}".format(layerIdx)
                serviceOutput.create_group(currentGroupName)

                if (layerIdx < len(self.outputGradientsWrtLayer)):
                    gradientsWrtInput, gradientsWrtOutput, gradientsWrtLoss, tensorSlice, loss = self.sess.run([self.layerGradientsWrtInput[idx], \
                            self.outputGradientsWrtLayer[layerIdx], self.lossGradientsWrtLayer[layerIdx], self.slicedTensors[idx], self.loss], \
                            feed_dict={self.inputPlaceholder: np.expand_dims(currentInput, axis=0), self.labelsPlaceholder: np.array(currentLabel)})

                    gradientsWrtOutput = removeBatchDim(gradientsWrtOutput)
                    gradientsWrtInput = removeBatchDim(gradientsWrtInput)
                    gradientsWrtLoss = removeBatchDim(gradientsWrtLoss)

                    if verbose:
                        print("Input grads shape:", gradientsWrtInput.shape)
                        print("Output grads shape:", gradientsWrtOutput.shape)

                    perFilterGradientWrtOutput = cummulateGradients(gradientsWrtOutput)
                    gradientsWrtOutput = normalizeValues(gradientsWrtOutput)

                    perFilterGradientWrtLoss = cummulateGradients(gradientsWrtLoss)

                    # Scale values
                    perFilterGradientWrtOutput = np.abs(perFilterGradientWrtOutput)  # For absolute scaling
                    perFilterGradientWrtOutput = normalizeValues(perFilterGradientWrtOutput)

                    if self.scaleGradientWrtLossValues:
                        perFilterGradientWrtLoss = np.abs(perFilterGradientWrtLoss)

                        # Scale values (gradient w.r.t. loss)
                        perFilterGradientWrtLoss = normalizeValues(perFilterGradientWrtLoss)

                    if verbose:
                        print("Filter gradient shape:", perFilterGradientWrtOutput.shape, "| Filter gradient value:", perFilterGradientWrtOutput)
                        print("Loss gradient shape:", perFilterGradientWrtLoss.shape, "| Loss gradient value:", perFilterGradientWrtLoss)

                    # Create the filter mask with percentile score
                    percentileIndex = int(np.round(perFilterGradientWrtOutput.shape[0] * self.currentPercentileValue))
                    sortedIndices = np.argsort(perFilterGradientWrtOutput)
                    percentileMask = np.zeros_like(perFilterGradientWrtOutput, dtype=np.bool)
                    percentileMask[sortedIndices[percentileIndex:]] = True

                    computeGradientWrtInputOnly = False

                else:
                    gradientsWrtInput = None
                    perFilterGradientWrtOutput = None
                    percentileMask = None

            if computeGradientWrtInputOnly:
                gradientsWrtInput, tensorSlice, loss, output = self.sess.run([self.layerGradientsWrtInput[idx], self.slicedTensors[idx], self.loss, self.outputLayer], \
                                feed_dict={self.inputPlaceholder: np.expand_dims(currentInput, axis=0), self.labelsPlaceholder: np.array(currentLabel)})

            # Create a new subplot for a new layer
            if (prevLayerName != currentLayerRootName):
                numSubPlotElements = math.ceil(math.sqrt(self.numLayerFilters[layerIdx]))
                plotIdxRow = 0
                plotIdxCol = 0
                layerIdx += 1

            else:
                plotIdxCol += 1
                if plotIdxCol == numSubPlotElements:
                    plotIdxRow += 1
                    plotIdxCol = 0

            prevLayerName = currentLayerRootName

            # Scale values
            gradientsWrtInput = np.abs(gradientsWrtInput)  # For absolute scaling
            saliencyMap = normalizeValues(gradientsWrtInput)
            saliencyMap = np.squeeze(saliencyMap)

            currentPlotIdx = (plotIdxRow * numSubPlotElements) + plotIdxCol

            # Transpose the values since the default size is [Rows, Cols]
            if len(gradientsWrtOutput.shape) > 1:
                filterSaliency = gradientsWrtOutput[:, currentPlotIdx].flatten()
            else:
                filterSaliency = gradientsWrtOutput[currentPlotIdx]

            filterImportance = perFilterGradientWrtOutput[currentPlotIdx] if perFilterGradientWrtOutput is not None else 1.0
            filterLossImpact = perFilterGradientWrtLoss[currentPlotIdx]
            filterMask = percentileMask[currentPlotIdx] if percentileMask is not None else True

            if verbose:
                layerName = self.layerNames[idx]
                print("Computing output for layer:", layerName)
                print("Saliency map shape:", saliencyMap.shape)

            currentSubGroupName = "{}/{}".format(
                currentGroupName, currentPlotIdx)
            serviceOutput.create_group(currentSubGroupName)

            serviceOutput[currentSubGroupName].create_dataset(
                "rawValue", data=tensorSlice.flatten())

            serviceOutput[currentSubGroupName].create_dataset(
                "saliencyMap", data=saliencyMap.T)

            serviceOutput[currentSubGroupName].create_dataset(
                "filterImportance", data=filterImportance)

            serviceOutput[currentSubGroupName].create_dataset(
                "filterSaliency", data=filterSaliency)

            serviceOutput[currentSubGroupName].create_dataset(
                "filterMask", data=bool(filterMask))

            serviceOutput[currentSubGroupName].create_dataset(
                "serviceOutput", data=float(filterLossImpact))

        # Remove redundant input-layer if pruned model is currently loaded
        if "input" in self.layerNames[0]:
            del serviceOutput['data/0']

        if fastMode != Modes.MINIMAL.value:
            # Iterate over all the filters to compute the rank of filter clusters
            # TODO: fix serviceOutput
            # computeFilterClusters(self, serviceOutput, visualizeFilters=visualizationFilters, verbose=verbose)
            pass

        if fastMode == Modes.FULL.value:
            print("Loss:", loss, "| Y:", np.squeeze(currentLabel), "| Output:", np.squeeze(output))

        serviceOutput.close()

        return serviceOutput


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~ Fliter Clustering ~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    def performFilterClustering(self):
        serviceOutput = self.loadData()
        computeFilterClusters(self, serviceOutput, visualizeFilters=False)


    def visualizeFilterClusters(self):
        serviceOutput = self.loadData(visualizationFilters=True)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~ Inverse Optimization ~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    def performInverseOptimization(self, y, iterations=1000, stepSize=1.0, randomStart=True, optimizeOnlyRawValues=True,
                                   smoothSampling=False, addSmoothnessPrior=False):
        inputPlaceholder = self.inputPlaceholder
        gtPlaceholder = self.labelsPlaceholder

        if randomStart:
            optimizedInput = np.random.normal(loc=0.0, scale=0.25, size=(1, self.trainX.shape[1], self.trainX.shape[2])) # Loc = Mean, Scale = Std. Dev.
            if smoothSampling:
                optimizedInput[0, :, 0] = smoothConsistentSampling(self.trainX.shape[1])
            elif addSmoothnessPrior:
                optimizedInput[0, :, 0] = smooth(optimizedInput[0, :, 0], 10)
        else:
            optimizedInput = np.zeros((1, self.trainX.shape[1], self.trainX.shape[2]))

        # Zero out other channels
        # if optimizeOnlyRawValues:
        #     for i in range(1, optimizedInput.shape[2]):
        #         optimizedInput[0, :, i] = np.zeros((self.trainX.shape[1]))

        # Get input saliency and forecast for the starting series
        startingSeries = optimizedInput.copy()
        startSerSaliencyMap = self.sess.run(self.outputGrad, feed_dict={self.inputPlaceholder: startingSeries})[0]
        startSerSaliencyMap = normalizeValues(np.abs(startSerSaliencyMap))
        startSerForecast = self.sess.run(self.model.layers[-1].output, feed_dict={inputPlaceholder: startingSeries})[0]

        for i in range(iterations):
            grad = self.sess.run(self.lossGrad, feed_dict={inputPlaceholder: optimizedInput, gtPlaceholder: y})
            grad = grad[0]

            # Subtract the gradient in order to decrease the loss
            if optimizeOnlyRawValues:
                optimizedInput[0, :, 0] -= (stepSize * grad[0, :, 0])
            else:
                optimizedInput -= (stepSize * grad)

        # Get input saliency and forecast for the optimized input
        saliencyMap = self.sess.run(self.outputGrad, feed_dict={self.inputPlaceholder: optimizedInput})[0]
        saliencyMap = normalizeValues(np.abs(saliencyMap))
        forecastValue = self.sess.run(self.model.layers[-1].output, feed_dict={inputPlaceholder: optimizedInput})[0]

        return startingSeries, startSerSaliencyMap, startSerForecast, optimizedInput, forecastValue, saliencyMap


    def performFGSMAttack(self, x, y, performIterativeAttack=False, useFGSM=True, iterations=100, stepSize=0.5,
                          alpha=1e-3, optimizeOnlyRawValues=True, numValuesToOptimize=None):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performFGSMTwoAttack(self, x, y, performIterativeAttack=True, useFGSM=True, iterations=100, stepSize=0.5,
                             alpha=1e-3, optimizeOnlyRawValues=True, numValuesToOptimize=None):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performIterativeFGSMAttack(self, x, y, performIterativeAttack=True, useFGSM=True, iterations=100, stepSize=0.5,
                                   alpha=1e-3, optimizeOnlyRawValues=True, numValuesToOptimize=None):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performGMAttack(self, x, y, performIterativeAttack=False, useFGSM=False, iterations=100, stepSize=10.0,
                        alpha=1e-1, optimizeOnlyRawValues=True, numValuesToOptimize=1):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performAttack(self, x, y, performIterativeAttack=False, useFGSM=False, iterations=100, stepSize=0.5, alpha=1e-3,
                      optimizeOnlyRawValues=True, numValuesToOptimize=None):
        inputPlaceholder = self.inputPlaceholder
        gtPlaceholder = self.labelsPlaceholder

        originalInput = np.copy(x)
        optimizedInput = x
        if optimizeOnlyRawValues:
            optimizedInput[0, :, 1] = np.zeros((self.trainX.shape[1]))

        for i in range(iterations):
            # grad = self.sess.run(gradient, feed_dict={inputPlaceholder: optimizedInput, gtPlaceholder: y})[0]
            [currentLoss, grad] = self.sess.run([self.loss, self.signGrad if useFGSM else self.lossGrad],
                                                feed_dict={inputPlaceholder: optimizedInput, gtPlaceholder: y})
            grad = grad[0]
            # print("Loss:", currentLoss) # Attack log

            # Clip the grad
            if numValuesToOptimize is not None:
                for channel in range(grad.shape[2]):
                    sortedIdx = np.argsort(grad[0, :, channel])[::-1]  # Sort descending
                    # topIdx = sortedIdx[:numValuesToOptimize]
                    bottomIdx = sortedIdx[numValuesToOptimize:]
                    grad[0, bottomIdx, channel] = 0.0

            if not performIterativeAttack:
                # Add the gradient in order to increase the loss
                if optimizeOnlyRawValues:
                    optimizedInput[0, :, 0] += (stepSize * grad[0, :, 0])
                else:
                    optimizedInput += (stepSize * grad)

                break

            # Add the gradient in order to increase the loss
            if optimizeOnlyRawValues:
                optimizedInput[0, :, 0] = np.clip(optimizedInput[0, :, 0] + (alpha * grad[0, :, 0]),
                                                  originalInput[0, :, 0] - stepSize, originalInput[0, :, 0] + stepSize)
            else:
                optimizedInput = np.clip(optimizedInput + (alpha * grad), originalInput - stepSize,
                                         originalInput + stepSize)

        forecastValue = self.sess.run(self.model.layers[-1].output, feed_dict={inputPlaceholder: optimizedInput})[0]

        # Get input saliency (assuming only one output)
        saliencyMap = self.sess.run(self.outputGrad, feed_dict={self.inputPlaceholder: optimizedInput})[0]
        saliencyMap = normalizeValues(np.abs(saliencyMap))

        return optimizedInput, forecastValue, saliencyMap


    def performInverseOptimizationAndAdvAttack(self):
        gtPoint = self.testY[self.inputIterator]
        startingSeries, startSerSaliencyMap, startingSeriesForecast, invOptimizedSeries, invOptimizedForecast, invOptSaliencyMap = \
            self.performInverseOptimization(gtPoint, stepSize=1e-2, smoothSampling=True, iterations=1000, optimizeOnlyRawValues=False)

        advExampleOrig, forecastValueAdvOrig, advExSaliencyMap = self.performIterativeFGSMAttack(
            np.copy(self.testX[self.inputIterator][np.newaxis, :, :]), gtPoint, alpha=1e-4, stepSize=1e-1,
            iterations=1000, optimizeOnlyRawValues=False)

        return startingSeries, startingSeriesForecast, startSerSaliencyMap, invOptimizedSeries, invOptimizedForecast, invOptSaliencyMap, advExampleOrig, forecastValueAdvOrig, advExSaliencyMap
