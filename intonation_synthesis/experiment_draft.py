import numpy
import tensorflow as tf
from matplotlib import cm
from PIL import Image, ImageFont, ImageOps

import const
import settings
from datasets import HTSDataset
from net import build_net
from train import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, pad_data)
from utils import list_nested_conv_layers


sess = tf.compat.v1.InteractiveSession()


def add_borders_to_images(images_list, border_size=(50, 100)):
    for index in range(len(images_list)):
        images_list[index] = ImageOps.expand(
            images_list[index], border=border_size, fill='white')


def get_concat_v(images_list, line_width=5):
    total_height = sum([img.height for img in images_list])
    dst = Image.new('RGB', (images_list[0].width, total_height), color="white")
    current_height = 0
    for image in images_list:
        dst.paste(image, (0, current_height))
        current_height += image.height + line_width
    return dst


with open('/opt/ml/model/training_report.txt', 'r') as report_file:
    validation_data = [line.strip("\n") for line in report_file.readlines()]
hts_validation_dataset = HTSDataset(
    settings.WORKDIR + 'input/data/training', file_list=validation_data,
    transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
    min_f0=const.MIN_F0, max_f0=const.MAX_F0,
    min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
    rich_feats=const.RICH_FEATS)

model = build_net(HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
model.load_weights("/opt/ml/model/model.hdf5")

tcn_layer = model.layers[0]

conv_layers = list_nested_conv_layers(tcn_layer._layers)
outputs = []
images = []

X, y = hts_validation_dataset[0]
X = X.astype("float32")
out = conv_layers[1](conv_layers[0](X)).eval()
outputs.append(out)

X_t = X[0].transpose()
out_t = out[0].transpose()
out_t /= numpy.max(numpy.abs(out_t), axis=0)

X_img = Image.fromarray(numpy.uint8(cm.bone(X_t) * 255))
out_img = Image.fromarray(numpy.uint8(cm.bone(out_t) * 255))
out_img = out_img.resize((X_img.width, X_img.height))
images.extend([X_img, out_img])

for layer_index in range(3, len(conv_layers)):
    out = conv_layers[layer_index](outputs[-1]).eval()
    outputs.append(out)
    out_t = out[0].transpose()
    out_t /= numpy.max(numpy.abs(out_t), axis=0)
    out_img = Image.fromarray(numpy.uint8(cm.bone(out_t) * 255))
    out_img = out_img.resize((X_img.width, X_img.height))
    images.append(out_img)


add_borders_to_images(images)
stacked_img = get_concat_v(images)
stacked_img.save("/app/results.png")


X_img_layer_outputs = Image.fromarray(numpy.uint8(cm.bone(X_t) * 255))
layer_output_images = []
X_pred = tcn_layer.call(X)
for layer_output in tcn_layer.layers_outputs:
    if not isinstance(layer_output, numpy.ndarray):
        layer_output = layer_output.eval()
    layer_output_t = layer_output[0].transpose()
    layer_output_t = numpy.max(numpy.abs(layer_output_t), axis=0)
    layer_out_img = Image.fromarray(numpy.uint8(cm.bone(layer_output_t) * 255))
    layer_out_img = layer_out_img.resize((
        X_img_layer_outputs.width, X_img_layer_outputs.height))
    layer_output_images.append(layer_out_img)

stacked_layer_out_img = get_concat_v(layer_output_images)
stacked_layer_out_img.save("/app/results_layers.png")


