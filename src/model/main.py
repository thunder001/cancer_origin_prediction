# #--------------------- Running in command line  -------------------
import argparse
import pandas as pd
from model_training import ModelTraining
from model_selection import run_model_selection
from model_evaluation import ModelEval
# import matplotlib
# matplotlib.use("TkAgg")
from plots import plot_pr, plot_auc


def main():
    args, _ = parser.parse_known_args()
    # print(args)
    parafile = args.parafile
    # print(parafile)

    train = ModelTraining()
    ev = ModelEval()
    paralist = pd.read_csv(parafile, sep="\t")
    print('\nInput parameters: {}\n'.format(paralist))
    para = dict(zip(paralist.key, paralist.value))

    if para['runtype'] == "train":
        trainfile = para['trainfile']
        units = int(para['units'])
        save_model_file = para['modelfile']
        print(save_model_file)

        train.run_training(trainfile, units, save_model_file)

    if para['runtype'] == "model_selection":
        trainfile = para['trainfile']
        testfile = para['testfile']
        units = para['units'].split(',')
        units = [int(unit) for unit in units]
        sample_size = int(para['sample_size'])
        model_dir = para['model_dir']
        best_model_dir = para['best_model_dir']

        run_model_selection(trainfile, testfile, units, sample_size, model_dir, best_model_dir)

    if para['runtype'] == "cv":

        CVData_dir = para['CVData_dir']
        CVModel_dir = para['model_dir']
        n_fold = int(para['folds'])
        sample_size = int(para['sample_size'])
        units = int(para['units'])
        codesfile = para['codesfile']
        results_dir = para['results_dir']

        SSPN, SSPN_stat = ev.cross_validation(CVData_dir, CVModel_dir, n_fold, sample_size, units, codesfile)
        print(SSPN)
        print(SSPN_stat)
        SSPN_stat.to_csv(results_dir + 'sspn_stat.csv')

    if para['runtype'] == 'test':

        testfile = para['testfile']
        testmetafile = para['testmetafile']
        sample_size = int(para['sample_size'])
        units = int(para['units'])
        modelfile = para['modelfile']
        # print(modelfile)
        # print(type(modelfile))
        codesfile = para['codesfile']
        figure_dir = para['figure_dir']
        results_dir = para['results_dir']

        accuracy, precision, recall, average_precision_ser, cm_df, SSPN_df, pred_df, fprs, tprs, aucs, tprs_overall = \
            ev.testeval(testfile, testmetafile, sample_size, units, modelfile, codesfile)

        # display metrics
        # print("\nAccuracy: {}".format(round(accuracy, 4)))
        print("\nConfusion matrix:")
        print(cm_df)
        print("\nAverage precision:")
        print(average_precision_ser)
        print("\nSSPN:\n {}".format(SSPN_df))

        # display precision-recall curve
        origins = list(average_precision_ser.index)
        # print(origins)

        # plt.figure()
        # plt.plot([1, 2, 3, 4, 5])
        # plt.show()

        # plt.plot([1, 2, 3, 4, 5], [1,2,3,4,5])

        # plt.savefig('./figures/text.png')

        # plt.show()

        # pr_fname = figure_dir + 'pr_' + parafile.split('/')[-1].split('.')[0] + '.png'
        # plot_pr(origins, precision, recall, average_precision_ser, codesfile, True, pr_fname)
        roc_fname = figure_dir + 'auc_' + parafile.split('/')[-1].split('.')[0] + '.png'
        plot_auc(origins, fprs, tprs, aucs, tprs_overall, codesfile, True, roc_fname)

        # save results to files

        # pd.Series(accuracy, index=['accuracy']).to_csv(results_dir + "accuracy.txt", sep="\n")
        # cm_df.to_csv(results_dir + "confusion_matrix.csv")
        # SSPN_df.to_csv(results_dir + "sspn.csv")
        # average_precision_ser.to_csv(results_dir + "average_precision.csv")
        # pred_df.to_csv(results_dir + "pred_results.csv", sep=",")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get performance of cancer origin prediction model using "
                                                 "test data")

    parser.add_argument(
        'parafile',
        help='A file containing parameters including running type and input files'
    )

    main()



# def main():
#     args, _ = parser.parse_known_args()
#     print(args)
#
#     train = ModelTraining()
#     ev = ModelEval()
#     if args.run_type == "train":
#         trainfile = args.trainfile
#         units = args.units[0]
#         save_model_file = args.modelfile
#         print(save_model_file)
#
#         train.run_training(trainfile, units, save_model_file)
#
#     if args.run_type == "model_selection":
#         trainfile = args.trainfile
#         testfile = args.testfile
#         units = args.units
#         sample_size = args.sample_size
#         model_dir = args.model_dir
#         best_model_dir = args.best_model_dir
#
#         run_model_selection(trainfile, testfile, units, sample_size, model_dir, best_model_dir)
#
#     if args.run_type == "cv":
#
#         CVData_dir = args.CVData_dir
#         CVModel_dir = args.model_dir
#         n_fold = args.folds
#         sample_size = args.sample_size
#         units = args.units[0]
#         codesfile = args.codesfile
#         results_dir = args.results_dir
#
#         SSPN, SSPN_stat = ev.cross_validation(CVData_dir, CVModel_dir, n_fold, sample_size, units, codesfile)
#         print(SSPN)
#         print(SSPN_stat)
#         SSPN_stat.to_csv(results_dir + 'sspn_stat.csv')
#
#     if args.run_type == "test":
#         testfile = args.testfile
#         testmetafile = args.testmetafile
#         sample_size = args.sample_size
#         units = args.units[0]
#         modelfile = args.modelfile
#         codesfile = args.codesfile
#         results_dir = args.results_dir
#
#         accuracy, precision, recall, average_precision_ser, cm_df, SSPN_df, pred_df = \
#             ev.testeval(testfile, testmetafile, sample_size, units, modelfile, codesfile)
#
#
#         # display metrics
#         # print("\nAccuracy: {}".format(round(accuracy, 4)))
#         print("\nConfusion matrix:")
#         print(cm_df)
#         print("\nAverage precision:")
#         print(average_precision_ser)
#         print("\n SSPN:\n {}".format(SSPN_df))
#
#         # save results to files
#
#         pd.Series(accuracy, index=['accuracy']).to_csv(results_dir + "accuracy.txt", sep="\n")
#         cm_df.to_csv(results_dir + "confusion_matrix.csv")
#         SSPN_df.to_csv(results_dir + "sspn.csv")
#         average_precision_ser.to_csv(results_dir + "average_precision.csv")
#         pred_df.to_csv(results_dir + "pred_results.csv", sep=",")
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description="Get performance of cancer origin prediction model using "
#                                                  "test data")
#     print(parser)
#
#     parser.add_argument(
#         'run_type',
#         choices=["train", "test", "cv", "model_selection"],
#         default='test',
#         help='Choose the type of program to run'
#     )
#
#     parser.add_argument(
#         '--trainfile',
#         nargs="?",
#         default="./Data/train_TCGA.tfrecords",
#         help='Methylation file as tfrecords to be used for training model'
#     )
#
#     parser.add_argument(
#         '--testfile',
#         nargs="?",
#         default="./Data/GEO.tfrecords",
#         help='Methylation file to be tested as tfrecords format'
#     )
#
#     parser.add_argument(
#         '--testmetafile',
#         nargs="?",
#         default="./Data/GEO_meta.csv",
#         help='Meta data file to be used in testing'
#     )
#
#     parser.add_argument(
#         '--modelfile',
#         nargs="?",
#         default="./Model/model_100.ckpt",
#         help='Model file to be used in testing or to be saved in training.'
#     )
#
#     parser.add_argument(
#         '--codesfile',
#         nargs="?",
#         default="./Data/code.csv",
#         help='Map file from cancer origin name to numeric value as csv format.'
#     )
#
#     parser.add_argument(
#         '--CVData_dir',
#         nargs="?",
#         default="./Data_test/CV_10/",
#         help='Directory for storing methylation data for each fold'
#     )
#     parser.add_argument(
#         '--model_dir',
#         nargs="?",
#         default="./Model/CV_model/",
#         help='Directory for storing models for each fold'
#     )
#
#     parser.add_argument(
#         '--units',
#         nargs="*",
#         type=int,
#         default=[64, 128, 256],
#         help='a list of hidden units to test'
#     )
#
#     parser.add_argument(
#         '--best_model_dir',
#         default="./Model/CV_model/",
#         help='Directory for best model'
#     )
#
#     parser.add_argument(
#         '--sample_size',
#         nargs="?",
#         type=int,
#         choices=[1468, 701, 581, 448, 431, 143],
#         help='Test sample size'
#     )
#     parser.set_defaults(sample_size=734)
#
#     parser.add_argument(
#         '-f', '--folds',
#         type=int,
#         default=10,
#         help='Number of folds to be generated'
#     )
#
#     parser.add_argument(
#         '--results_dir',
#         nargs="?",
#         default="./DNN_results/DNN_100_dev/",
#         help='folder for test results'
#     )
#
#     main()

# -------------------------Command line script---------------------------------------------

# model training
# python3 ./python/main.py    train \
#                             --trainfile ./DNN_data/train1.tfrecords \
#                             --units 64 \
#                             --modelfile ./DNN_model/train_model/train.ckpt \
#

# model selection
# python3 ./python/main.py      model_selection \
#                                 --trainfile ./DNN_data/train1.tfrecords \
#                                 --testfile ./DNN_data/dev.tfrecords \
#                                 --units 64 128 256 \
#                                 --sample_size 1468 \
#                                 --model_dir ./DNN_model/models/ \
#                                 --best_model_dir ./DNN_model/best_model_2/ \


# cross validation performance
# python3 ./python/main.py    cv \
#                             --CVData_dir ./Cancer_origin_prediction/data/CV/ \
#                             --model_dir ./DNN_model/cv_model_2/ \
#                             --units 64  \
#                             --sample_size 431 \
#                             --codesfile ./DNN_data/code.csv \
#                             --results_dir ./DNN_results/CV/

# test set performance
# python3 ./python/main.py test \
#                                         --testfile ./DNN_data/test1.tfrecords \
#                                         --testmetafile ./DNN_data/test1_meta.csv \
#                                         --sample_size 1468 \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/code.csv \
#                                         --results_dir ./DNN_results/test/


# metastatic cancer performance
# python3 ./python/main.py test \
#                                         --testfile ./DNN_data/metastatic_data2.tfrecords \
#                                         --testmetafile ./DNN_data/metastatic_data2_meta.csv \
#                                         --sample_size 143 \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/code.csv \
#                                         --results_dir ./DNN_results/metastatic2/


# GEO set performance
# python3 ./python/cancer_origin_DNN_2.py test \
#                                         --testfile ./Cancer_origin_prediction/data/GEO/combined_final.tfrecords \
#                                         --testmetafile ./Cancer_origin_prediction/data/GEO/combined_final_meta.csv \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/code.csv \
#                                         --sample_size 581 \
#                                         --results_dir ./DNN_results/GEO/

