import argparse
import numpy as np
import os
import pandas as pd
import random
import sklearn.model_selection as ms
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def csv_to_tfrecords(inputfile):

    """Converts a dataset (stored in CSV file) to tfrecords.
    step 1: split data into train, test sets
    step 2: convert into TFRecords by getting features, labels, stuff into an example,
    then serialized into a string
    """
    dat = pd.read_csv(inputfile).values
    print(dat.shape)
    outputfile = inputfile.split(".")[0] + ".tfrecords"
    with tf.python_io.TFRecordWriter(outputfile) as writer:
        for row in dat:
            features, label = row[:-1], row[-1]
            example = tf.train.Example()
            example.features.feature["features"].float_list.value.extend(features)
            example.features.feature["label"].int64_list.value.append(int(label))
            writer.write(example.SerializeToString())


def train_test_split(df, df_meta, train_frac=0.80, seed=0):
    """
    Random split data into training and test dataset according to
    assigned ratio.

    """
    df.index = range(df.shape[0])
    index = list(df.index)
    random.seed(seed)
    random.shuffle(index)
    cutpoint = int((len(index) * train_frac))
    print(cutpoint)
    train_index = index[:cutpoint]
    print(max(train_index))
    test_index = index[cutpoint:]

    train = df.iloc[train_index, :]
    test = df.iloc[test_index, :]

    train_meta = df_meta.iloc[train_index, :]
    test_meta = df_meta.iloc[test_index, :]
    return train, test, train_meta, test_meta


def train_dev_test_split(df, df_meta, train_frac=0.80, dev_frac=0.1, seed=1):
    """
    Random split data into training and test dataset according to
    assigned ratio.

    """
    df.index = range(df.shape[0])
    index = list(df.index)
    random.seed(seed)
    random.shuffle(index)
    devpoint = int((len(index) * train_frac))
    testpoint = int(len(index) * (train_frac + dev_frac))
    print(devpoint)
    print(testpoint)
    train_index = index[:devpoint]
    dev_index = index[devpoint: testpoint]
    test_index = index[testpoint:]

    train = df.iloc[train_index, :]
    dev = df.iloc[dev_index, :]
    test = df.iloc[test_index, :]

    train_meta = df_meta.iloc[train_index, :]
    dev_meta = df_meta.iloc[dev_index, :]
    test_meta = df_meta.iloc[test_index, :]
    return train, dev, test, train_meta, dev_meta, test_meta


def getANOVA(cpg, groups, train_dat):
    """
    ANOVA test for methylation of each CpG cite among all cancer origin groups

    :param cpg: each column in dataset
    :param groups: cancer origin groups in dataset
    :param train_dat: a dataframe for training dataset
    :return: one dimension p value array [number of CpG cites]
    """
    mod = ols('cpg ~ groups', data=train_dat).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    p_value = np.round(aov_table.iloc[0, 3], 4)
    return p_value


def getMaxDiff(cpg, groups):
    """
    Obtain maximal difference beta value of methylation for each CpG cites among all groups

    :param cpg: each column in dataset
    :param groups: cancer origin groups in dataset
    :return: maximal difference beta value of methylation
    """
    mc = MultiComparison(cpg, groups)
    result = mc.tukeyhsd()
    maxdiff = np.max(np.abs(result.meandiffs))
    return maxdiff


def getSub(df, df_meta, num_of_sample):
    """
    Obtain subset of data that satisfy minimal cases for each cancer origins

    :param df: raw cancer origin dataset
    :param num_of_sample: minimal number of cases
    :return: subset of data
    """
    # Caution when dealing with meta data
    # sample_counts = df.primary_site_code.value_counts() # uncomment when input is all_labeled_data
    sample_counts = df.primary_site.value_counts()
    samples = sample_counts[sample_counts > num_of_sample]
    # df_sub = df.loc[df.primary_site_code.isin(samples.index), :] # uncomment when input is all_labeled_data
    df_sub = df.loc[df.primary_site.isin(samples.index), :]
    df_meta_sub = df_meta.loc[df.primary_site.isin(samples.index), ]
    return df_sub, df_meta_sub


def data_prep(inputfile, inputfile_meta, num_of_case, outdir, dev=True):
    """
    Prepare training and test dataset from input data

    :param inputfile: input datafile as csv format
    :param inputfile_meta: input metadata file as csv format
    :param num_of_case: minimal number of cases
    :param outdir: Output data folder
    :param dev: boolean to set if having development data set
    :return: training and test dataset
    """

    # Step 1: preprocessing: read data from file, get subset,
    # handle missing values and label data using numeric value
    print("\nReading data...")
    dat = pd.read_csv(inputfile)
    dat_meta = pd.read_csv(inputfile_meta)

    print("\nProcessing input data...")
    dat, dat_meta = getSub(dat, dat_meta, num_of_case)

    # handle missing values in column
    thresh = np.round(dat.shape[0] * 0.1)
    dat.dropna(axis=1, thresh=thresh, inplace=True)
    dat.fillna(dat.mean(), inplace=True)

    # convert character primary_site into numeric type
    # primary_sites = dat.primary_site_name.unique() # uncomment when input is all_labeled_data
    primary_sites = dat.primary_site.unique()
    primary_sites.sort()
    primary_site_code = range(np.alen(primary_sites))
    site_code = dict(zip(primary_sites, primary_site_code))

    site_code_df = pd.Series(site_code)
    codefile = outdir + "code.csv"
    site_code_df.to_csv(codefile)

    # dat['primary_site_code'] = dat.primary_site_name.map(site_code)
    dat['primary_site_code'] = dat.primary_site.map(site_code)

    # Step 2: Train/dev/test split
    if dev:
        train_dat, dev_dat, test_dat, train_meta, dev_meta, test_meta = train_dev_test_split(dat, dat_meta)
    else:
        train_dat, test_dat, train_meta, test_meta = train_test_split(dat, dat_meta)

    print("\nPreparing train data...")
    # step 3: train data processing - feature selection - Anova and turkey
    train_dat_X = train_dat.iloc[:, :-2]
    train_dat_y = train_dat.iloc[:, -1]
    groups = train_dat.primary_site_code

    p_values = train_dat_X.apply(getANOVA, groups=groups, train_dat=train_dat)
    max_diffs = train_dat_X.apply(getMaxDiff, groups=groups)

    train_dat_X = train_dat_X.loc[:, p_values < 0.01]
    train_dat_X = train_dat_X.loc[:, max_diffs > 0.15]
    features = train_dat_X.columns
    print("\nTotal features: {}".format(features.shape[0]))
    feafile = outdir + 'features.txt'
    fw = open(feafile, 'w')
    for item in features.values:
        fw.write(item + '\n')

    train = pd.concat([train_dat_X, train_dat_y], axis=1)

    # save data to csv file and tfrecords
    trainfile = outdir + "train1.csv"
    train.to_csv(trainfile, index=False)
    # convert csv to tfrecords
    csv_to_tfrecords(trainfile)
    trainmetafile = outdir + "train1_meta.csv"
    train_meta.to_csv(trainmetafile, index=False)


    # Step 4: Dev/Test data processing
    # Feature filtering according to selection features in train data
    if dev:
        print("\nPreparing dev data...")
        dev_dat_X = dev_dat.loc[:, dev_dat.columns.isin(features)]
        dev_dat_y = dev_dat.iloc[:, -1]
        dev = pd.concat([dev_dat_X, dev_dat_y], axis=1)
        # Save data to csv file
        devfile = outdir + "dev.csv"
        dev.to_csv(devfile, index=False)
        # convert csv to tfrecords
        csv_to_tfrecords(devfile)
        devmetafile = outdir + "dev_meta.csv"
        dev_meta.to_csv(devmetafile, index=False)

        print("\nPreparing test data...")
        test_dat_X = test_dat.loc[:, test_dat.columns.isin(features)]
        test_dat_y = test_dat.iloc[:, -1]
        test = pd.concat([test_dat_X, test_dat_y], axis=1)
        # Save data to csv file
        testfile = outdir + "test1.csv"
        test.to_csv(testfile, index=False)
        # convert csv to tfrecords
        csv_to_tfrecords(testfile)
        testmetafile = outdir + "test1_meta.csv"
        test_meta.to_csv(testmetafile, index=False)
        print("\nAll done!")
    else:
        print("\nPreparing test data...")
        test_dat_X = test_dat.loc[:, test_dat.columns.isin(features)]
        test_dat_y = test_dat.iloc[:, -1]
        test = pd.concat([test_dat_X, test_dat_y], axis=1)
        # Save data to csv file
        testfile = outdir + "test1.csv"
        test.to_csv(testfile, index=False)
        # convert csv to tfrecords
        csv_to_tfrecords(testfile)
        testmetafile = outdir + "test1_meta.csv"
        test_meta.to_csv(testmetafile, index=False)
        print("\nAll done!")


def test_prep(inputfile_test, inputmetafile, codesfile, feafile, outdir):
    """
    Prepare test dataset from input file

    :param inputfile_test: input data file as csv format
    :param inputmetafile: input meta data file as csv format
    :param codesfile: a map file from cancer origin to numeric value
    :param feafile: a txt file containg feature columns
    :param testfile_output: file path for processed test data
    :param testmetafile_output: file path for processed meta data
    :return: processed test data and test metadata as tfrecords format
    """

    print("\nReading feature and label data...")
    features = pd.read_csv(feafile, sep="\n", header=None)[0].values
    codes = pd.read_csv(codesfile, header=None)
    codes = dict(zip(codes[0], codes[1]))
    labels = list(codes.keys())

    print("\nReading test data...")
    test = pd.read_csv(inputfile_test)
    test_label = test['primary_site'].map(codes)

    print('Test labels: {}'.format(test_label.values))
    print(type(test_label))
    #
    # print("\nHandling missing values...")
    #
    test = test.loc[test.primary_site.isin(labels), test.columns.isin(features)]
    print(test.shape)
    # print("\nTest features: {}".format(len(list(test.columns))))
    if test.isna().any().sum() != 0:
        test.fillna(test.mean(), inplace=True)

    # handling missing features in test set
    test_fea = list(test.columns)

    if len(test_fea) != len(features):
        print("\nHandling missing features...")
        mis_fea = set(features) - set(test_fea)  # find missed features
        print(mis_fea)
        # generate a random dataframe for missed features
        mis_df = pd.DataFrame(np.random.rand(test.shape[0], len(mis_fea)), columns=mis_fea)
        test = pd.concat([test, mis_df], axis=1)  # obtain complete test set
        test.sort_index(axis=1, inplace=True) # sort the column

    test['primary_code'] = test_label

    print("\nWriting to CSV...")
    filename1 = inputfile_test.split("/")[-1]
    test_file = outdir + filename1
    test.to_csv(test_file, index=False)
    print("\nConverting to tfrecords...")
    csv_to_tfrecords(test_file)

    print("\nWriting meta data to CSV...")
    test_meta = pd.read_csv(inputmetafile)
    test_meta = test_meta.loc[test_meta.primary_site.isin(labels), ]

    filename2 = inputmetafile.split('/')[-1]
    testmeta_file = outdir + filename2
    test_meta.to_csv(testmeta_file, index=False)


def test_prep_folder(testdat_dir, codesfile, feafile, outdir):
    files = os.listdir(testdat_dir)

    datfiles = [file for file in files if 'meta' not in file]
    datfiles = [testdat_dir + file for file in datfiles]
    metafiles = [file for file in files if 'meta' in file]
    metafiles = [testdat_dir + file for file in metafiles]

    for file, metafile in zip(datfiles, metafiles):
        test_prep(file, metafile, codesfile, feafile, outdir)


def CV_prep(datafile, folds_folder, folds):
    """
    Prepare n fold cross validation data

    :param datafile: input datafile for split
    :param folds_folder: cross-validation data folder
    :param folds: number of folds to be splitt
    :return: n fold data sets as tfrecords format
    """

    dat = pd.read_csv(datafile)
    labels = dat.iloc[:, -1]

    skf = ms.StratifiedKFold(folds, shuffle=True, random_state=0).split(dat, labels)

    fold = 0
    for train_index, test_index in skf:
        print("Processing fold {0} of {1}...".format(fold + 1, folds))
        train_set = dat.iloc[train_index, :]
        test_set = dat.iloc[test_index, :]

        train_file = folds_folder + "train_" + str(fold) + ".csv"
        test_file = folds_folder + "test_" + str(fold) + ".csv"

        train_set.to_csv(train_file, index=False)
        test_set.to_csv(test_file, index=False)

        csv_to_tfrecords(train_file)
        csv_to_tfrecords(test_file)
        fold += 1


# ------------------------------------------ Running in command line -----------------------------
def main():
    args, _ = parser.parse_known_args()
    print(args)

    if args.prep_type == "train_test":
        inputfile = args.datafile
        inputmetafile = args.metadatafile

        num_of_case = 100
        outdir = args.outdir

        data_prep(inputfile, inputmetafile, num_of_case, outdir, dev=False)

    if args.prep_type == "train_dev_test":
        inputfile = args.datafile
        inputmetafile = args.metadatafile

        num_of_case = 100
        outdir = args.outdir

        data_prep(inputfile, inputmetafile, num_of_case, outdir)

    if args.prep_type == "test":

        inputfile_test = args.datafile
        inputmetafile_test = args.metadatafile
        codesfile = args.codesfile
        feafile = args.featurefile
        outdir = args.outdir

        test_prep(inputfile_test, inputmetafile_test,codesfile, feafile, outdir)

    if args.prep_type == "test_folder":

        datdir = args.testdir
        codesfile = args.codesfile
        feafile = args.featurefile
        outdir = args.outdir
        test_prep_folder(datdir, codesfile, feafile, outdir)

    if args.prep_type == "cv":

        datafile = args.datafile
        outdir = args.outdir
        n_fold = args.folds

        CV_prep(datafile, outdir, n_fold)


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


# -------------------------IDE running-------------------------------
# inputfile = "/Users/zhengc/Projects/cancer_origin/TCGA/DNN_data/all_labeled_data1.csv"
# # inputfile_meta = "/Users/zhengc/Projects/cancer_origin/TCGA/DNN_data/all_labeled_data1_meta.csv"
# inputfile_meta = "/Users/zhengc/Projects/cancer_origin/TCGA/DNN_data/all_labeled_data1_meta.csv"
#
# num_of_case = 100
# outdir = "/Users/zhengc/Projects/cancer_origin/TCGA/DNN_data/DNN_100_data/train_dev_test/"
#
# data_prep(inputfile, inputmetafile, num_of_case, outdir)

# -------------------Command line script-----------------------------
# train_dev_test prep
# python3 ./python/data_prep_2.py train_dev_test \
#                                         --datafile ./DNN_data/all_labeled_data1.csv \
#                                         --metadatafile ./DNN_data/all_labeled_data1_meta.csv \
#                                         --outdir ./DNN_data/DNN_100_data/train_dev_test_15_10/

# independent set prep
# python3 ./python/data_prep_2.py test_folder \
#                                     --testdir ../GEO1/  \
#                                     --featurefile ./DNN_data/DNN_100_data/train_dev_test_15/features.txt\
#                                     --codesfile   ./DNN_data/DNN_100_data/train_dev_test_15/code.csv\
#                                     --outdir ../GEO/test_data_100360/

# metastatic data set prep
# python3 ./python/data_prep_2.py test \
#                                     --datafile ./DNN_data/metastatic_data1.csv  \
#                                     --metadatafile ./DNN_data/metastatic_data1_meta.csv\
#                                     --codesfile   ./DNN_data/DNN_100_data/train_dev_test_15_20/code.csv\
#                                     --featurefile ./DNN_data/DNN_100_data/train_dev_test_15_20/features.txt \
#                                     --outdir ./DNN_data/DNN_100_data/metastatic_data_15_20/


# cross-validation set prep
# python3 ./python/data_prep_2.py cv \
#                                     --datafile ./DNN_data/DNN_100_data/train_dev_test_15_20/train1.csv \
#                                     --outdir ./DNN_data/DNN_100_data/cv_data_15_20/








