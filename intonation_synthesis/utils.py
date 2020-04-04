import numpy
import os
import pandas
import tensorflow as tf

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


def _get_layer_of_nth_path_in_residual_block(res_block, n, layer_name):
    results = sorted(
        [
            (layer, layer.name) for layer in res_block.layers
            if layer_name in layer.name
        ],
        key=lambda x: int(
            x[0].name[x[0].name.rfind("_") + 1:]
            if x[0].name[x[0].name.rfind("_") + 1:].isnumeric() else 0))

    print(results[0][1])
    return results[0][0]

def list_nested_conv_layers(layers_list, skip_suffix=None):
    parsed_layers = []
    for layer in layers_list:
        if not isinstance(layer, tf.keras.layers.Layer):
            continue
        elif "conv" in layer.name:
            if skip_suffix is not None:
                if layer.name.endswith(skip_suffix):
                    continue
            filters, biases = layer.get_weights()
            print("{}: filters: {}".format(layer.name, filters.shape))
            parsed_layers.append(layer)
        elif "residual" in layer.name:
            print("Residual Block: {}:".format(layer.name))
            conv = _get_layer_of_nth_path_in_residual_block(
                layer, 0, "conv1D")
            norm = _get_layer_of_nth_path_in_residual_block(
                layer, 0, "batch_normalization")
            activation = _get_layer_of_nth_path_in_residual_block(
                layer, 0, "activation")
            dropout = _get_layer_of_nth_path_in_residual_block(
                layer, 0, "spatial_dropout1d")
            residual_layers_first_path = [conv, norm, activation, dropout]
            parsed_layers.extend(residual_layers_first_path)
        elif "lambda" in layer.name:
            continue
        else:
            continue
            print("Layer: {} (skipping)".format(layer.name))

    return parsed_layers


def get_tcn_model_layers(model):
    tcn_layer = model.layers[0]
    # camelCase notation copied from the TSViz-Core library
    outputLayers = list_nested_conv_layers(tcn_layer._layers)
    outputLayers.extend(model.layers[1:])
    return outputLayers
