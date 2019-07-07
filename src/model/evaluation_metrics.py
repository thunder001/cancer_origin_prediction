import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from scipy import interp
from collections import OrderedDict
from sklearn.metrics import precision_recall_curve, average_precision_score, \
    confusion_matrix, auc, roc_curve

# np.set_printoptions(threshold=np.inf)


class Metrics(object):

    def make_one_hot(self, labels, num_classes):
        """
        Make one hot matrix from class labels

        :param labels: numeric class labels
        :param num_classes: number of classes
        :return: numpy array - [num of labels, num_classes]
        """

        """
        Make one hot matrix from class labels
        Note: need to assign class number
        """
        a = np.zeros([labels.size, num_classes])
        a[np.arange(labels.size), labels] = 1
        return a

    def acc(self, preds, labels):
        """
        Evaluate the overall accuracy of prediction
        :param preds: prediction class
        :param labels: true lable class
        :return: accuracy value
        """
        correct = [1 if preds[i] == labels[i] else 0 for i in range(len(preds))]
        return np.mean(correct, dtype=np.float32)

    def SSPN(self, conf_matrix_df):
        """ Compute both class wise SSPN and overall SSPN based on multiple class confusion matrix
        Note: SSPPA stands for specificity, sensitivity, positive predictive values, negative
        predictive value

        :param: conf_matrix_df: Pandas DataFrame - [number of prediction classes, number of prediction classes
        :return SSPN_df: Pandas DataFrame - [number of labelled classes, 5]
        """

        conf_matrix = conf_matrix_df.values
        # print(conf_matrix)
        SSPN = []
        SSPN_index = []

        classes = conf_matrix_df.index.values
        n_class = len(classes)

        for i in range(n_class):
            class_count = sum(conf_matrix[i, :])
            if class_count != 0:   # !!Just compute SSPN for labelled classed
                SSPN_index.append(classes[i])
                TP = conf_matrix[i, i]
                FP = sum(conf_matrix[:, i]) - TP
                FN = sum(conf_matrix[i, :]) - TP
                TN = np.sum(conf_matrix) - FP - FN - TP

                sp = TN / (TN + FP)
                sn = TP / (TP + FN)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)
                # acc = (TP + TN) / (TP + FP + TN + FN) #// not necessary for multiple classes
                # SSPN.append([sp, sn, ppv, npv, acc])
                SSPN.append([sp, sn, ppv, npv])

        SSPN_array = np.array(SSPN)

        overall = np.mean(SSPN_array, axis=0)
        SSPN_index.append('Overall')

        SSPN_array = np.vstack((SSPN_array, overall))
        SSPN_df = pd.DataFrame(SSPN_array,
                                index=SSPN_index,
                                columns=['Specificity', 'Sensitivity', 'PPV', 'NPV'])

        return SSPN_df

    def pr(self, logs_soft, labels):
        """
        Compute class wise and overall precisions and recalls
        :param logs_soft: softmax values of DNN model outputs
        :param labels: True sample labels
        :return: three dictionary objects that stores precisions, recalls
        and average precisions
        """

        precision = dict()
        recall = dict()
        average_precision = dict()

        labs_hot = self.make_one_hot(labels, 18)
        for i in range(labs_hot.shape[1]):
            # get precision recall and average precision for individual class
            if labs_hot[:, i].sum() != 0:   # test if dataset contain this class
                precision[i], recall[i], _ = precision_recall_curve(labs_hot[:, i], logs_soft[:, i])
                average_precision[i] = average_precision_score(labs_hot[:, i], logs_soft[:, i], average="micro")

        # get overall precision recall and average precision
        precision[labs_hot.shape[1]], recall[labs_hot.shape[1]], _ = \
            precision_recall_curve(labs_hot.ravel(), logs_soft.ravel())
        average_precision[labs_hot.shape[1]] = average_precision_score(labs_hot, logs_soft, average="micro")

        return precision, recall, average_precision

    def auc(self, logs_soft, labels):
        """
        Compute class wise and overall fpr and tpr
        :param logs_soft: softmax values of DNN model outputs
        :param labels: True sample labels
        :return: four objects that stores fprs, tprs, aucs and overall tprs
        """

        fprs = dict()
        tprs = dict()
        aucs = dict()
        tprs_overall = []

        mean_fpr = np.linspace(0, 1, 100)

        labs_hot = self.make_one_hot(labels, 18)
        for i in range(labs_hot.shape[1]):
            # get fpr and tpr for individual class
            if labs_hot[:, i].sum() != 0:   # test if dataset contain this class
                fprs[i], tprs[i], thresholds = roc_curve(labs_hot[:, i], logs_soft[:, i])
                tprs_overall.append(interp(mean_fpr, fprs[i], tprs[i]))
                tprs_overall[-1][0] = 0.0
                roc_auc = auc(fprs[i], tprs[i])
                aucs[i] = roc_auc

        return fprs, tprs, aucs, tprs_overall

    def mean_conf(self, SSPN, conf=0.95):
        Mean = SSPN.mean(axis=1)
        Std = SSPN.sem(axis=1)
        h = Std * sp.stats.t.ppf((1 + conf) / 2., SSPN.shape[1] - 1)
        CI_low = Mean - h
        CI_up = Mean + h
        SSPN_stat = pd.DataFrame(OrderedDict({'Mean': Mean, 'Std': Std,
                                              'CI_low': CI_low, 'CI_up': CI_up}))
        SSPN_stat.index = ['Specificity', 'Sensitivity',
                           'Postive_predictive_value', 'Negative_predictive_value', 'Accuracy']
        return SSPN_stat

    def evaluate(self, labels, logs_soft, preds, codesfile):

        # 1. Accuracy
        accuracy = self.acc(preds, labels)

        # 2. Average_precision
        precision, recall, average_precision = self.pr(logs_soft, labels)

        codes = pd.read_csv(codesfile, header=None)
        codes_map = dict(zip(codes[1], codes[0]))
        code_overall = len(codes_map)
        codes_map[code_overall] = "Overall"  # !Note: add overall precision to mapping dictionary

        average_precision_ser = pd.Series(average_precision)
        average_precision_ser.index = average_precision_ser.index.map(lambda x: codes_map[x])
        average_precision_ser.dropna(inplace=True)

        # 3. ROC
        fprs, tprs, aucs, tprs_overall = self.auc(logs_soft, labels)

        # 4. Multiple class confusion matrix
        # Note: always square matrix, dimension is class number of true labels or predicted labels,
        # which is bigger
        cm = confusion_matrix(labels, preds)
        num_label_class = np.unique(labels).size
        num_pred_class = np.unique(preds).size

        if num_label_class > num_pred_class:
            test_class = np.unique(labels).tolist()
        else:
            test_class = np.unique(preds).tolist()

        # test_class = np.unique(labels).tolist()
        test_class.sort()
        test_index = [codes_map[x] for x in test_class]
        # print(test_index)
        cm_df = pd.DataFrame(cm, index=test_index, columns=test_index)

        # 5. SSPN matrix
        SSPN_df = self.SSPN(cm_df)

        return accuracy, precision, recall, average_precision_ser, cm_df, SSPN_df, fprs, tprs, aucs, tprs_overall

    def predict(self, labels, preds, codesfile, testmetafile):

        codes = pd.read_csv(codesfile, header=None)
        codes_map = dict(zip(codes[1], codes[0]))
        code_overall = len(codes_map)
        codes_map[code_overall] = "Overall"  # !Note: add overall precision to mapping dictionary

        labs_name = list(map(lambda x: codes_map[x], labels))  # Note: map function returns a map object
        # print(len(labs_name))
        preds_name = list(map(lambda y: codes_map[y], preds))
        # print(len(preds_name))

        correct = []
        for i in range(len(labs_name)):
            if labs_name[i] == preds_name[i]:
                correct.append('True')
            else:
                correct.append('False')

        meta = pd.read_csv(testmetafile)
        if list(meta.columns).__contains__('barcode'):
            patient = list(meta['patient'])
            diagnosis = list(meta['name'])

            pred_df = pd.DataFrame(OrderedDict({'Patient': patient, 'Diagnosis': diagnosis, 'Primary_site': labs_name,
                                                'Prediction': preds_name, 'Correct': correct}))

        else:
            geo_id = list(meta['geo_accession'])
            pred_df = pd.DataFrame(OrderedDict({'GEO_ID': geo_id, 'Primary_site': labs_name,
                                                'Prediction': preds_name, 'Correct': correct}))

        return pred_df
