
import pandas as pd
import matplotlib as mpl
import datetime
import logging

import matplotlib.pyplot as plt

import networkx as nx
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def set_plot_style():
    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 200

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['legend.fontsize'] = 'medium'
    mpl.rcParams['figure.titlesize'] = 'medium'


def read_data(data_path):
    # read documentation
    file1 = data_path + "/sensor_info.csv"
    info = pd.read_csv(file1, index_col=0)

    # read data
    file2 = data_path + "/data_unscaled_clean.csv"
    data = pd.read_csv(file2, index_col=0, parse_dates=True)

    # check data
    assert (data.index.is_monotonic)
    assert (data.index.nunique() == len(data.index))
    assert (len(info) == len(data.columns))

    data.columns = data.columns.astype(int)

    # only consider first 163 sensors
    # N = 163
    N = len(data.columns)
    data = data[data.columns[0:N]]
    info = info.loc[range(N)]

    return data, info


def show_progress(now, elapsed_time, ratio, in_hour=False):
    print("\n")
    print("now = " + str(now))
    print("status = " + "{:.2f}".format(100 * ratio) + " %")
    print("elapsed time (in seconds) = ", "{:.2f}".format(elapsed_time))
    if in_hour:
        print("estimated total time (in hours) = ", "{:.2f}".format((elapsed_time / ratio) / 3600))
    else:
        print("estimated total time (in seconds) = ", "{:.2f}".format((elapsed_time / ratio)))


def log_config(log_path):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(filename=log_path + '/' + cur_time + ".log",
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    return logging


def log_progress(now, logging, elapsed_time, ratio):
    # log status
    logging.info("\n" +
                 "now = " + str(now) + "\n" +
                 "status = " + "{:.2f}".format(100 * ratio) + " %" + "\n" +
                 "elapsed time (in seconds) = " + "{:.2f}".format(elapsed_time) + "\n" +
                 "estimated total time (in hours) = " + "{:.2f}".format((elapsed_time / ratio) / 3600) + "\n")


def visualize_adjacent_graph(A, now=None):
    # visualize A
    G = nx.Graph(A)
    pos = nx.spring_layout(G, k=10 / np.sqrt(len(A)), iterations=100)
    #pos = nx.spring_layout(G)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    if now is not None:
        plt.title("Timestamp: " + str(now))

    nx.draw_networkx(G, pos,
                     with_labels=True, node_color='gray',
                     edgelist=edges, edge_color=weights,
                     node_size=200, font_size=8,
                     alpha=0.8,
                     edge_vmin=min(weights), edge_vmax=max(weights),
                     width=1.0, edge_cmap=plt.cm.Reds)

    plt.gcf().set_size_inches(10, 8)
    plt.gcf().set_dpi(200)
    # plt.tight_layout()

    plt.axis('off')


def visualize_adjacent_heatmap(A, now=None):
    plt.figure(figsize=(10, 8))
    if now is not None:
        plt.title("Timestamp: " + str(now))
    sns.heatmap(A, annot=False, cmap='Reds')
    plt.xlabel("sensor index")
    plt.ylabel("sensor index")
    plt.tight_layout()
    plt.gcf().set_dpi(200)


def visualize_node_event(X, now=None):
    plt.figure(figsize=(12, 2))
    if now is not None:
        plt.title("Timestamp: " + str(now))
    sns.heatmap(X.transpose(), vmin=-1, vmax=1, cmap='coolwarm', cbar_kws={"ticks": [-1, 0, 1]})
    plt.tight_layout()
    plt.gcf().set_dpi(200)


def visualize_node_feature(X, now=None):
    plt.figure(figsize=(12, 2))
    vlim = max(3, abs(X).max())
    if now is not None:
        plt.title("Timestamp: " + str(now))
    sns.heatmap(X.transpose(), vmin=-vlim, vmax=vlim, cmap='coolwarm')
    plt.tight_layout()
    plt.gcf().set_dpi(200)


def visualize_anomaly_score(X, now=None):
    plt.figure(figsize=(12, 2))
    vlim = abs(X).max()
    if now is not None:
        plt.title("Timestamp: " + str(now))
    sns.heatmap(X.transpose(), vmin=0, vmax=vlim, cmap='Reds')
    plt.tight_layout()
    plt.gcf().set_dpi(200)


def visualize_anomaly_label(X, now=None):
    plt.figure(figsize=(12, 2))
    if now is not None:
        plt.title("Timestamp: " + str(now))
    sns.heatmap(X.transpose(), vmin=0, vmax=1, cmap='Reds')
    plt.tight_layout()
    plt.gcf().set_dpi(200)


def get_roc_auc(y_true, y_score, output_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    print("save roc data")
    roc_data = np.array([fpr, tpr, thresholds]).T
    roc_data = pd.DataFrame(roc_data, columns=['fpr', 'tpr', 'thresholds'])
    roc_data['auc'] = auc(fpr, tpr)
    roc_data.to_csv(output_path + "/roc_data.csv")


def plot_roc_auc(y_true, y_score, output_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if output_path is not None:

        print("save roc data")
        roc_data = np.array([fpr, tpr, thresholds]).T
        roc_data = pd.DataFrame(roc_data, columns=['fpr', 'tpr', 'thresholds'])
        roc_data['auc'] = auc(fpr, tpr)
        roc_data.to_csv(output_path + "/roc_data.csv")

    plt.figure(figsize=(6, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def get_precision_recall(y_true, y_score, output_path):
    average_precision = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) < len(precision):
        thresholds = np.hstack((thresholds, thresholds[-(len(precision) - len(thresholds)):]))

    assert (np.isnan(precision).sum() == 0)
    assert (np.isnan(recall).sum() == 0)
    assert (np.isnan(thresholds).sum() == 0)

    eps = 1e-16
    f1_arr = 2 * precision * recall / (precision + recall + eps)

    assert (np.isnan(f1_arr).sum() == 0)

    print("save prc data")
    prc_data = np.array([precision, recall, thresholds, f1_arr]).T
    prc_data = pd.DataFrame(prc_data, columns=['precision', 'recall', 'thresholds', 'f1_score'])
    prc_data['average_precision'] = average_precision
    prc_data.to_csv(output_path + "/prc_data.csv")


def plot_precision_recall(y_true, y_score, output_path=None):
    average_precision = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) < len(precision):
        thresholds = np.hstack((thresholds, thresholds[-(len(precision) - len(thresholds)):]))

    assert (np.isnan(precision).sum() == 0)
    assert (np.isnan(recall).sum() == 0)
    assert (np.isnan(thresholds).sum() == 0)

    eps = 1e-16
    f1_arr = 2 * precision * recall / (precision + recall + eps)

    assert (np.isnan(f1_arr).sum() == 0)

    if output_path is not None:
        print("save prc data")
        prc_data = np.array([precision, recall, thresholds, f1_arr]).T
        prc_data = pd.DataFrame(prc_data, columns=['precision', 'recall', 'thresholds', 'f1_score'])
        prc_data['average_precision'] = average_precision
        prc_data.to_csv(output_path + "/prc_data.csv")

    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, where='post', label='precision recall curve')
    plt.plot(recall, f1_arr, label='f1 score recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision={0:0.2f}'.format(
        average_precision))
    plt.axis('equal')
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

