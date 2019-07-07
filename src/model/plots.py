import pandas as pd
import matplotlib
import numpy as np
from sklearn.metrics import auc
from itertools import cycle
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_pr(origins, precision, recall, average_precision_ser, codesfile, each):
    codes = pd.read_csv(codesfile, header=None)
    codes_map = dict(zip(codes[1], codes[0]))
    codes_map[len(codes_map)] = "Overall"  # !Note: add overall precision to mapping dictionary
    # print("codes_map:\n {}".format(codes_map))
    origin_map = dict(zip(codes[0], codes[1]))
    origin_map['Overall'] = len(origin_map)
    # print("origin_map:\n {}".format(origin_map))
    classes = [origin_map[origin] for origin in origins]

    average_precision_ser.index = average_precision_ser.index.map(lambda x: origin_map[x])
    average_precision = average_precision_ser.to_dict()
    # print(average_precision)

    # print(codes_map)

    plt.figure(figsize=(7, 8))

    if each:

        lines = []
        labels = []

        # print(recall[1])
        # print(precision[1])

        for cls in classes:
            # print(cls)
            if cls != 18:
                line, = plt.plot(recall[cls], precision[cls], lw=1)
                plt.plot(recall[cls], precision[cls], lw=1, alpha=0.3)
                lines.append(line)
                labels.append('{0:s} (MAP = {1:0.2f})'.format(codes_map[cls], average_precision[cls]))
            else:
                line, = plt.plot(recall[cls], precision[cls], color='b', lw=1)
                plt.plot(recall[cls], precision[cls], lw=2, color='b', alpha=0.8)
                lines.append(line)
                labels.append('{0:s} (MAP = {1:0.2f})'.format(codes_map[cls], average_precision[cls]))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', size=20)
        plt.ylabel('Precision', size=20)
        # plt.title('Precision-Recall curve for each cancer type')
        plt.legend(lines, labels, loc='best', prop=dict(size=12))

    else:
        # -------------Plot overall precision-recall curve-----------
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall', size=20)
        plt.ylabel('Precision', size=20)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                  .format(average_precision['micro']))

    plt.savefig('./figures/pr_geo.png')


def plot_auc(origins, fprs, tprs, aucs, tprs_overall, codesfile, each):
    codes = pd.read_csv(codesfile, header=None)
    codes_map = dict(zip(codes[1], codes[0]))
    # print("codes_map:\n {}".format(codes_map))
    origin_map = dict(zip(codes[0], codes[1]))
    # print("origin_map:\n {}".format(origin_map))
    classes = [origin_map[origin] for origin in origins if origin != 'Overall']

    plt.figure(figsize=(10, 7))

    if each:

        lines = []
        labels = []

        for cls in classes:
            line, = plt.plot(fprs[cls], tprs[cls], lw=1)
            plt.plot(fprs[cls], tprs[cls], lw=1, alpha=0.3)
            lines.append(line)
            labels.append('{0:s} (AUC = {1:0.2f})'.format(codes_map[cls], aucs[cls]))

        # line2, = plt.plot(recall[n_class - 1], precision[n_class - 1], color='blue', lw=3)
        # plt.plot(recall[n_class - 1], precision[n_class - 1], color='blue', lw=3)
        # lines.append(line2)
        # labels.append('Overall Precision-recall (MAP = {0:0.2f})'.format(average_precision[n_class - 1]))

        # print(labels2)

        # fig = plt.gcf()
        # fig.subplots_adjust(bottom=0.25)

        # ax = plt.gca()
        # ax.plot("recall", "precision", "-o", data=pr11_df)
        # ax.set_xlabel("Recall", fontsize=14)
        # ax.set_ylabel("Precision", fontsize=14)
        # ax.axis([0, 1.05, 0, 0.1])

        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs_overall, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        # print("AUCs:\n {}".format(aucs))
        # print("FPRS:\n {}".format(fprs))
        # print("TPRS:\n {}".format(tprs))
        # print(list(aucs.values()))
        std_auc = np.std(list(aucs.values()))
        # print(mean_auc)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        line, = plt.plot(mean_fpr, mean_tpr, lw=1, color='b')
        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8)
        lines.append(line)
        labels.append(r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate', size=20)
        plt.ylabel('True positive rate', size=20)
        # plt.title('ROC curves for metastatic samples', size=16)
        plt.legend(lines, labels, loc='best', prop=dict(size=12))

    plt.savefig('./figures/auc_test.png')

