
import argparse
from data import Data


def main():
    args, _ = parser.parse_known_args()
    print(args)

    if args.prep_type == "train_test":
        inputfile = args.datafile
        inputmetafile = args.metadatafile

        num_of_case = 100
        outdir = args.outdir

        Data.train_dev_test_prep(inputfile, inputmetafile, num_of_case, outdir, dev=False)

    if args.prep_type == "train_dev_test":
        inputfile = args.datafile
        inputmetafile = args.metadatafile

        num_of_case = 100
        outdir = args.outdir

        Data.train_dev_test_prep(inputfile, inputmetafile, num_of_case, outdir)

    if args.prep_type == "test":

        inputfile_test = args.datafile
        inputmetafile_test = args.metadatafile
        codesfile = args.codesfile
        feafile = args.featurefile
        outdir = args.outdir

        Data.test_prep(inputfile_test, inputmetafile_test,codesfile, feafile, outdir)

    if args.prep_type == "test_folder":

        datdir = args.testdir
        codesfile = args.codesfile
        feafile = args.featurefile
        outdir = args.outdir
        Data.test_prep_folder(datdir, codesfile, feafile, outdir)

    if args.prep_type == "cv":

        datafile = args.datafile
        outdir = args.outdir
        n_fold = args.folds

        Data.CV_prep(datafile, outdir, n_fold)


if __name__ == '__main__':

    """
    command line interface
    """
    parser = argparse.ArgumentParser(description="Data preparation for DNN model training and evaluation")

    parser.add_argument(
        'prep_type',
        choices=["train_test", "train_dev_test", "test", "test_folder","cv"],
        default="test",
        help='Choose the type of program to run'
    )

    parser.add_argument(
        '--datafile',
        nargs="?",
        default="./Data/metastatic_TCGA.csv",
        help='Methylation file to be processed'
    )


    parser.add_argument(
        '--metadatafile',
        nargs="?",
        default="./Data/metastatic_TCGA_meta.csv",
        help='Corresponding meta data file'
    )

    parser.add_argument(
        '--testdir',
        nargs="?",
        default="./Data/metastatic_TCGA.csv",
        help='folder for test data'
    )

    parser.add_argument(
        '--featurefile',
        nargs="?",
        default="./Model/features.txt",
        help='Feature file'
    )

    parser.add_argument(
        '--codesfile',
        nargs="?",
        default="./Model/code.csv",
        help='Codes map file from cancer origin to numeric value'
    )

    parser.add_argument(
        '--outdir',
        nargs="?",
        default="./Data_test/",
        help='Output data folder.'
    )

    parser.add_argument(
        '-f', '--folds',
        type=int,
        default=10,
        help='Number of folds to be generated'
    )

    args, _ = parser.parse_known_args()
    print(args)

    main()


# ------------------------------------------ Running in command line -----------------------------

# train_dev_test prep
# python3 ./python/data/main.py train_dev_test \
#                                         --datafile ./DNN_data/all_labeled_data1.csv \
#                                         --metadatafile ./DNN_data/all_labeled_data1_meta.csv \
#                                         --outdir ./DNN_data/DNN_100_data/train_dev_test_15_10/

# independent set prep
# python3 ./python/data/main.py test_folder \
#                                     --testdir ../GEO1/  \
#                                     --featurefile ./DNN_data/DNN_100_data/train_dev_test_15/features.txt\
#                                     --codesfile   ./DNN_data/DNN_100_data/train_dev_test_15/code.csv\
#                                     --outdir ../GEO/test_data_100360/

# metastatic data set prep
# python3 ./python/data/main.py test \
#                                     --datafile ./DNN_data/metastatic_data2.csv  \
#                                     --metadatafile ./DNN_data/metastatic_data2_meta.csv\
#                                     --codesfile   ./DNN_data/code.csv\
#                                     --featurefile ./DNN_data/features.txt \
#                                     --outdir ./DNN_data/metastatic/


# cross-validation set prep
# python3 ./python/data/main.py cv \
#                                     --datafile ./DNN_data/train1.csv \
#                                     --outdir ./DNN_data/CV/

