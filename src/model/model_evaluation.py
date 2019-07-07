
import pandas as pd
from model_training import ModelTraining
from model_testing import run_testing
from evaluation_metrics import Metrics


class ModelEval(object):

    @staticmethod
    def testeval(testfile, testmetafile, sample_size, units, modelfile, codesfile):

        """ Run testing and return performance metrics

        :param testfile tfrecords formated testing file
        :param modelfile saved model for testing
        :param sample_size sample size for test data
        :param codesfile a csv file coding for cancer origins

        :return performance metrics, including precision, recall, average_precision,
        SSPN, comfusion matrix and overall accuracy
        """
        ev = Metrics()
        labels, logs_soft, preds, labs_hot, acc = run_testing(testfile, modelfile, units, sample_size)

        accuracy, precision, recall, average_precision_ser, cm_df, SSPN_df, fprs, tprs, aucs, tprs_overall  = \
            ev.evaluate(labels, logs_soft, preds, codesfile)

        pred_df = ev.predict(labels, preds, codesfile, testmetafile)

        return accuracy, precision, recall, average_precision_ser, \
               cm_df, SSPN_df, pred_df, fprs, tprs, aucs, tprs_overall

    @staticmethod
    def cross_validation(CVfiles_folder, CVModel_folder, folds, sample_size, units, codesfile):
        ev = Metrics()
        training = ModelTraining()
        SSPN_overall = []

        for i in range(folds):
            print("\n\nRunning fold {0} of {1} ...".format(i+1, folds))
            trainfile = CVfiles_folder + "train_" + str(i) + ".tfrecords"
            print(trainfile)
            testfile = CVfiles_folder + "test_" + str(i) + ".tfrecords"
            print(testfile)
            CVModelfile = CVModel_folder + "model_" + str(i) + ".ckpt"
            print(CVModelfile)

            training.run_training(trainfile, units, CVModelfile)
            labels, logs_soft, preds, labs_hot, acc = run_testing(testfile, CVModelfile, units, sample_size)

            accuracy, _, _, _, _, SSPN_df = \
                ev.evaluate(labels, logs_soft, preds, codesfile)

            overall = list(SSPN_df.loc["Overall"])
            overall.append(accuracy)
            # print("Overall {}".format(overall))
            SSPN_overall.append(overall)
            # print("SSPN Overall {}".format(SSPN_overall))

        SSPN_overall = pd.DataFrame(SSPN_overall, columns=['Specificity', 'Sensitivity',
                                                           'Postive_predictive_value', 'Negative_predictive_value',
                                                           'Accuracy'])
        SSPN_overall = SSPN_overall.transpose()
        SSPN_stat = ev.mean_conf(SSPN_overall).round(4)

        return SSPN_overall, SSPN_stat
