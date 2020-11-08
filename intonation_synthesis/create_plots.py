import innvestigate
import json
import matplotlib.gridspec as grd
import numpy as np
import os
import pandas as pd
import re
import matplotlib
from matplotlib import colors, pyplot as plt, rc, ticker

import const
import settings
from datasets import HTSDataset
from feature_names import FEATURE_GROUPS, DETAILED_GROUPS, ALL_GROUPS
from net import build_net
from train import pad_data


RESULTS_PATH = '/opt/ml/model'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

MODEL_PATH = '/app/model/model.hdf5'
TRAINING_REPORT_PATH = '/app/model/training_report.txt'
DATASET_PATH = '/opt/ml/input/data/training/data/txt/'
HYPERPARAMETERS_PATH = os.path.join(settings.WORKDIR, 'input/config', 'hyperparameters.json')

# This should point to a directory containing a list of reference natural f0 (in Hz)
REFERENCE_F0_DIR_PATH = '/other/5ms/f0'

# Plotting consts
LINTHRESH = 3
LINSCALE = 0.001
LOGSTEP = 1


with open(HYPERPARAMETERS_PATH) as json_file:
    hyperparameters = json.load(json_file)

BATCH_SIZE = int(hyperparameters.get('batch_size'))
HIDDEN_SIZE = int(hyperparameters.get('hidden_size'))
NUM_LAYERS = int(hyperparameters.get('num_layers'))
DROPOUT = float(hyperparameters.get('dropout'))


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


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


def make_log_ticks(linthresh, logstep, vmin, vmax):
    tick_locations = [0.0]
    if vmax > 0.0:
        maxlog = int(np.ceil(np.log10(vmax)))
        tick_locations = tick_locations + [(10.0 ** x) for x in np.arange(-linthresh + logstep, maxlog + 1, logstep)]
    if vmin < 0.0:
        minlog = int(np.ceil(np.log10(-vmin)))
        tick_locations = ([-(10.0 ** x) for x in np.arange(-linthresh + logstep, minlog + 1, logstep)][::-1]
                          + tick_locations)

    return tick_locations

def plot_colormesh(ax, analysis, linthresh, linscale, limits, cmap='PiYG'):

    vmin, vmax = limits

    if not vmin < 0.0:
        cmap = 'Greens'
    elif not vmax > 0.0:
        cmap = 'Greens_r'
    pcm = ax.pcolormesh(
        analysis, norm=colors.SymLogNorm(
            10**-linthresh, linscale=linscale,
            vmin=float(vmin), vmax=float(vmax)),
        cmap=cmap)

    return pcm


def make_plots(analysis, y, preds, filename, linthresh=LINTHRESH, linscale=LINSCALE, logstep=LOGSTEP, title=None):
    f0_values_start_idx, f0_values_end_idx = np.where(y)[0][[0, -1]]

    fig = plt.figure()
    gs = grd.GridSpec(2, 2, height_ratios=[3, 6], width_ratios=[6, 1], wspace=0.1)
    ax1 = plt.subplot(gs[2])

    vmin = analysis.min()
    vmax = analysis.max()

    pcm = plot_colormesh(ax1, analysis, linthresh, linscale, (vmin, vmax))

    # generate logarithmic ticks
    tick_locations = make_log_ticks(linthresh, logstep, vmin, vmax)

    plt.xlabel('Sample number (time)')
    plt.ylabel('Feature index')
    plt.xlim(0, f0_values_end_idx + f0_values_start_idx)

    color_ax = plt.subplot(gs[3])
    cb = plt.colorbar(pcm, cax=color_ax, ticks=tick_locations, format=ticker.LogFormatterMathtext())
    cb.set_label('log-normalized LRP.Z')

    ax2 = plt.subplot(gs[0], sharex=ax1)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    ax2.plot(preds, 'k', zorder=5, lw=0.5, label=r'Predicted F0')
    ax2.plot(y, 'k', linestyle=':', alpha=0.3, zorder=0, lw=0.5, label='Ground truth')
    ax2.set_ylabel('Log F0')
    plt.xlim(0, f0_values_end_idx + f0_values_start_idx)
    plt.ylim(0, 5)

    legend = ax2.legend(loc='lower left')

    if title is not None:
        plt.suptitle(title)

    output_filename = filename.split('.')[0] + '.png'

    fig.savefig(os.path.join(RESULTS_PATH, output_filename), dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close('all')


def _plot_group_results_df(df, title):
    fig, ax = plt.subplots()
    plt.errorbar(df['group name'], df['group mean'], df['group std per feat'], marker='s', linestyle='none')
    plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=4)
    plt.suptitle(title)
    filename = "_".join(title.lower().split())
    plt.savefig(os.path.join(RESULTS_PATH, filename), bbox_inches='tight', pad_inches=0.5, dpi=600)
    plt.close('all')

def plot_group_results(df, title=None):
    _plot_group_results_df(df, title)
    if df.iloc[0]['group name'] == 'VUV':
        # V/UV usually has a mean value higher by orders of magnitude and flattens plots of other features
        # we plot again without it
        df_no_vuv = df.iloc[1:]
        _plot_group_results_df(df_no_vuv, title + " - no VUV")


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


def read_bin_file(filename, dir=REFERENCE_F0_DIR_PATH, dataformat='<f4'):
    fh = open(os.path.join(dir, filename))
    datatype = np.dtype('<f4')
    data = np.fromfile(fh, datatype)
    return data


def save_preds_to_bin_file(preds, filepath, data_format='<f4'):
    assert len(preds.shape) == 1
    with open(filepath, 'wb') as preds_fh:
        datatype = np.dtype(data_format)
        preds = preds.astype(datatype)
        preds.tofile(preds_fh, '')


def calculate_errors(preds, y):
    return {
        'MSE': mse(y, preds),
        'MAE': mae(y, preds),
        'RMSE': rmse(y, preds),
        'NRMSE': nrmse(y, preds),
    }


def perform_analysis():
    hts_validation_dataset = setup_data()
    model = setup_model()
    analyzer = setup_analyzer(model)
    summed_vector = None
    abs_summed_vector = None
    pos_summed_vector = None
    neg_summed_vector = None

    all_prediction_errors = []

    file_errors = []

    for index, (X, y) in enumerate(hts_validation_dataset):
        filename = hts_validation_dataset.data[index].split(".")[0] + ".f0"
        print("Analysing {}...".format(filename))
        analysis = analyzer.analyze(X)
        preds = model.predict(X).flatten()

        print("Calculating prediction errors...")
        current_errors = calculate_errors(preds, y.flatten())
        current_errors['file'] = filename

        all_prediction_errors.append(current_errors)

        # Trim prediction array to the size of the original input to get rid of the zero-padded tail
        data = read_bin_file(filename)
        if data.shape[0] < np.where(preds)[0][[0, -1]][1]:
            file_errors.append(filename)
        preds = preds[:data.shape[0]]
        # Get F0 from Log(F0)
        preds_freq = np.array([np.exp(val) if val != 0 else val for val in preds])

        filepath = os.path.join(RESULTS_PATH, filename)
        print("Saving prediction results {}...".format(filepath))
        save_preds_to_bin_file(preds_freq, filepath)

        txt_label_filename = hts_validation_dataset.data[index].split('.')[0].split('_')[-1] + '.txt'
        txt_label_path = os.path.join(DATASET_PATH, txt_label_filename)
        with open(txt_label_path, 'r', encoding='cp1250') as txt_label_fh:
            txt_label = u'"{}"'.format(" ".join(txt_label_fh.readlines()).strip())


        print("Plotting analysis results...")
        make_plots(
            analysis[0].T, y.flatten(), preds, hts_validation_dataset.data[index], title=txt_label)

        if index == 0:
            summed_vector = analysis[0]
            abs_summed_vector = np.abs(analysis[0])
            pos_summed_vector = analysis[0].clip(min=0)
            neg_summed_vector = analysis[0].clip(max=0)
        else:
            summed_vector += analysis[0]
            abs_summed_vector += np.abs(analysis[0])
            pos_summed_vector += analysis[0].clip(min=0)
            neg_summed_vector += analysis[0].clip(max=0)

    print('Calculating general statistics...')

    pd.DataFrame(all_prediction_errors).set_index('file').to_csv(os.path.join(RESULTS_PATH, 'all_prediction_errors.csv'))

    with open(hts_validation_dataset.question_file_name) as qst_file:
        questions = qst_file.readlines()

    feature_names = [
        re.sub("[\t\n]", "", re.sub('\"(.*)\"', "\g<1>", line))
        for line in questions if line.strip()
    ]
    feature_names.append('vuv')

    with open('log.txt', 'w') as log_fh:
        log_fh.writelines("\n".join(file_errors))

    summed_vectors = {
        'sum': summed_vector,
        'absolute sum': abs_summed_vector,
        'positive values sum': pos_summed_vector,
        'negative values sum': neg_summed_vector,
        'mean (sum)': summed_vector / len(hts_validation_dataset),
        'mean (absolute sum)': abs_summed_vector / len(hts_validation_dataset),
        'mean (positive only sum)': pos_summed_vector / len(hts_validation_dataset),
        'mean (negative only sum)': neg_summed_vector / len(hts_validation_dataset),
        'count': len(hts_validation_dataset),
    }

    pd.DataFrame([{'index': index, 'feature name': feat} for index, feat in enumerate(feature_names)]).set_index(
        'index').to_csv(os.path.join(RESULTS_PATH, 'feature_names.csv'), sep=",")

    return summed_vectors, all_prediction_errors, feature_names


def make_group_results_for_feature_groups(feature_ranking, feature_groups):

    group_individual_scores = {}

    for feat, score in feature_ranking:
        feat_group_names = [group_name for group_name, group_features in feature_groups.items() if feat in flatten(group_features)]
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
        group_std_per_feat = np.std(scores_list)
        group_var = np.var(scores_list)

        group_scores.append(
            {
                'group name': group_name,
                'group sum': group_sum,
                'group mean': group_mean,
                'group std per feat': group_std_per_feat,
                'group var': group_var,
            }
        )

    return group_scores


def make_results(return_results=False):

    summed_vectors, prediction_errors, feature_names = perform_analysis()

    if return_results:
        results = {
            'summed vectors': summed_vectors,
            'errors': prediction_errors,
            'feature names': feature_names,
        }

    for key, analysis in summed_vectors.items():
        if key == "count":
            continue

        plt.suptitle("Dataset {}".format(key))

        fig = plt.figure()
        gs = grd.GridSpec(1, 2, width_ratios=[6, 1], wspace=0.1)
        ax1 = plt.subplot(gs[0])
        vmin, vmax = analysis.min(), analysis.max()

        pcm = plot_colormesh(
            ax1, analysis.T, linthresh=LINTHRESH, linscale=LINSCALE, limits=(vmin, vmax))
        tick_locations = make_log_ticks(LINTHRESH, LOGSTEP, vmin, vmax)
        plt.xlabel('Sample number (time)')
        plt.ylabel('Feature index')
        ax2 = plt.subplot(gs[1])
        cb = plt.colorbar(pcm, cax=ax2, ticks=tick_locations, format=ticker.LogFormatterMathtext())
        cb.set_label('log-normalized LRP.Z.')

        fig.savefig(os.path.join(RESULTS_PATH, "{}.png".format("_".join(key.lower().split()))), bbox_inches='tight', pad_inches=0.5, dpi=600)
        plt.close("all")

        feature_relevance_ranking = sorted(
            [(summed_vectors[key].sum(axis=0)[index], index) for index in range(summed_vectors[key].shape[1])],
                key = lambda x: x[0], reverse = True)
        feature_relevance_ranking = [(feature_names[feat[1]], feat[0]) for feat in feature_relevance_ranking]

        general_results = make_group_results_for_feature_groups(
            feature_relevance_ranking, FEATURE_GROUPS)
        detailed_results = make_group_results_for_feature_groups(
            feature_relevance_ranking, DETAILED_GROUPS)
        all_groups_results = make_group_results_for_feature_groups(
            feature_relevance_ranking, ALL_GROUPS)

        results_dict = {
            "general": general_results,
            "detailed": detailed_results,
            "all": all_groups_results,
        }

        if return_results:
            results[key] = results_dict

        for detail_level in results_dict.keys():
            if 'negative' in key:
                ascending = True
            else:
                ascending = False
            df = pd.DataFrame(results_dict[detail_level]).sort_values(['group mean'], ascending=ascending)
            dataframe_title = "Feature relevancy ranking - {} ({})".format(key, detail_level)
            filename = "_".join(dataframe_title.lower().split()) + '.csv'
            plot_group_results(df, title=dataframe_title)
            df = df.set_index('group name')
            df.to_csv(os.path.join(RESULTS_PATH, filename), sep=',')

    if return_results:
        return results


if __name__ == "__main__":
    results = make_results()
    # TODO: Convert all dict values to some non-numpy json-serializable format
    # with open(os.path.join(RESULTS_PATH, 'results.json'), 'w') as results_fh:
        # print("Saving total results to JSON file...")
        # json.dump(results, results_fh, sort_keys=True, indent=4)