import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
matplotlib.use("TkAgg")


def plot_pr(n_class, precision, recall, average_precision, codesfile, each):

    codes = pd.read_csv(codesfile, header=None)
    codes_map = dict(zip(codes[1], codes[0]))
    codes_map[n_class - 1] = "Overall"  # !Note: add overall precision to mapping dictionary
    print(codes_map)
    if each:

        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.figure(figsize=(7, 8))

        # f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels2 = []
        # for f_score in f_scores:
        #     x = np.linspace(0.01, 1)
        #     y = f_score * x / (2 * x - f_score)
        #     l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        #     plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        #
        # lines.append(l)
        # labels2.append('iso-f1 curves')
        l, = plt.plot(recall[n_class - 1], precision[n_class - 1], color='blue', lw=3)
        lines.append(l)
        labels2.append('Overall Precision-recall (MAP = {0:0.2f})'
                       ''.format(average_precision[n_class - 1]))

        for i, color in zip(range(n_class - 1), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=1.5)
            lines.append(l)
            labels2.append('{0:s} (MAP = {1:0.2f})'
                           ''.format(codes_map[i], average_precision[i]))

        # fig = plt.gcf()
        # fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve for each cancer type')
        plt.legend(lines, labels2, loc='best', prop=dict(size=10))

    else:
        # -------------Plot overall precision-recall curve-----------
        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                  .format(average_precision['micro']))


