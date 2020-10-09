import innvestigate
import json
import matplotlib.gridspec as grd
import numpy as np
import os
import re
from matplotlib import colors, pyplot as plt, ticker

import const
import settings
from datasets import HTSDataset
from feature_names import FEATURE_GROUPS
from net import build_net
from train import pad_data


RESULTS_PATH = './results/'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

MODEL_PATH = '/app/model/model.hdf5'
TRAINING_REPORT_PATH = '/app/model/training_report.txt'
HYPERPARAMETERS_PATH = os.path.join(settings.WORKDIR, 'input/config', 'hyperparameters.json')


with open(HYPERPARAMETERS_PATH) as json_file:
    hyperparameters = json.load(json_file)

BATCH_SIZE = int(hyperparameters.get('batch_size'))
HIDDEN_SIZE = int(hyperparameters.get('hidden_size'))
NUM_LAYERS = int(hyperparameters.get('num_layers'))
DROPOUT = float(hyperparameters.get('dropout'))


def flatten(nested_list):
     for element in nested_list:
         if isinstance(element, list):
             yield from flatten(element)
         else:
             yield element


def get_bbox(array):
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    return (cmin, cmax, rmin, rmax)


def make_plots(analysis, y, preds, filename, norm='symlog', linthresh=0.001, linscale=0.001):
    f0_values_start_idx, f0_values_end_idx = np.where(y)[0][[0, -1]]
    fig = plt.figure()
    gs = grd.GridSpec(2, 2, height_ratios=[3, 6], width_ratios=[6, 1], wspace=0.1)
    ax1 = plt.subplot(gs[2])

    vmin = analysis.min()
    vmax = analysis.max()
    pcm = ax1.pcolormesh(
        analysis, norm=colors.SymLogNorm(
            linthresh=linthresh, linscale=0.001,
            vmin=vmin, vmax=vmax),
        cmap='binary')

    plt.xlabel('time')
    plt.ylabel('feature')
    plt.xlim(0, f0_values_end_idx + f0_values_start_idx)

    #generate logarithmic ticks
    max_log = int(np.ceil(np.log10(vmax)))
    min_log = int(np.ceil(np.log10(-vmin)))
    logstep = 1
    tick_locations=([-(10**x) for x in np.arange(-linthresh, min_log + 1, logstep)][::-1]
                    +[0.0]
                    +[(10**x) for x in np.arange(-linthresh, max_log+1, logstep)])
    import ipdb; ipdb.set_trace()
    pass

    color_ax = plt.subplot(gs[3])
    cb = plt.colorbar(pcm, cax=color_ax, ticks=tick_locations)
    cb.set_label('log-normalized LRP.Z')

    ax2 = plt.subplot(gs[0])

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.plot(preds, 'k', zorder=5, lw=0.5, label=r'Predicted F_0')
    ax2.plot(y, 'k', linestyle=':', alpha=0.3, zorder=0, lw=0.5, label='Ground truth')
    plt.xlim(0, f0_values_end_idx + f0_values_start_idx)
    plt.ylim(0, 5)

    legend = ax2.legend(loc='lower left')

    output_filename = filename.split('.')[0] + '.png'

    import ipdb; ipdb.set_trace()
    pass
    fig.savefig(os.path.join(RESULTS_PATH, output_filename), dpi=600)
    plt.close('all')


def setup_data():
    with open(TRAINING_REPORT_PATH, 'r') as report_file:
        validation_data = \
            [line.strip("\n") for line in report_file.readlines()]

    hts_validation_dataset = HTSDataset(
        settings.WORKDIR + 'input/data/training',
        file_list=validation_data,
        transform=pad_data, f0_backward_window_len=const.F0_WINDOW_LEN,
        min_f0=const.MIN_F0, max_f0=const.MAX_F0,
        min_duration=const.MIN_DURATION, max_duration=const.MAX_DURATION,
        rich_feats=const.RICH_FEATS)

    return hts_validation_dataset


def setup_model():
    model = build_net(HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model.load_weights(MODEL_PATH)
    model.trainable = False
    return model


def setup_analyzer(model):
    return innvestigate.create_analyzer('lrp.z', model, allow_lambda_layers=True, neuron_selection_mode="all")


def perform_analysis():
    hts_validation_dataset = setup_data()
    model = setup_model()
    analyzer = setup_analyzer(model)
    summed_vector = None

    for index, (X, y) in enumerate(hts_validation_dataset):
        print("Analysing {}...".format(hts_validation_dataset.data[index]))
        analysis = analyzer.analyze(X)
        preds = model.predict(X).flatten()
        make_plots(analysis[0].T, y.flatten(), preds, hts_validation_dataset.data[index])
        if summed_vector is None:
            summed_vector = analysis[0]
        else:
            summed_vector += analysis[0]

    with open(hts_validation_dataset.question_file_name) as qst_file:
        questions = qst_file.readlines()

    feature_names = [
        re.sub("[\t\n]", "", re.sub('\"(.*)\"', "\g<1>", line))
        for line in questions if line.strip()
    ]
    feature_names.append('vuv')

    return summed_vector, feature_names


summed_vector, feature_names = perform_analysis()

feature_relevance_ranking = sorted(
    [(np.abs(summed_vector.sum(axis=0))[index], index) for index in range(summed_vector.shape[1])],
        key = lambda x: x[0], reverse = True)


feature_ranking = [(feature_names[feat[1]], feat[0]) for feat in feature_relevance_ranking]

group_individual_scores = {}

for feat, score in feature_ranking:
    feat_group_names = [group_name for group_name, group_features in FEATURE_GROUPS.items() if feat in flatten(group_features)]
    if len(feat_group_names) == 0:
        import ipdb; ipdb.set_trace()
        pass
    for feat_group_name in feat_group_names:
        if group_individual_scores.get(feat_group_name):
            group_individual_scores[feat_group_name].append((feat, score))
        else:
            group_individual_scores[feat_group_name] = [(feat, score), ]


group_scores = []

for group_name, results_list in group_individual_scores.items():
    scores_list = [score for feat, score in results_list]
    no_scores = len(scores_list)
    group_sum = np.sum(scores_list)
    group_mean = group_sum/no_scores
    group_std_of_mean_per_feat = np.std(scores_list)
    group_var = np.var(scores_list)

    group_scores.append(
        {
            'group_name': group_name,
            'group_sum': group_sum,
            'group_mean': group_mean,
            'group_std_of_mean_per_feat': group_std_of_mean_per_feat,
            'group_var': group_var,
        }
    )


