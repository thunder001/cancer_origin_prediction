import pandas as pd
import sklearn.model_selection as ms
import os
from utility import Utility


class Data(object):

    @staticmethod
    def train_dev_test_prep(inputfile, inputfile_meta, num_of_case, outdir, dev=True):
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
        dat, dat_meta = Utility.getSub(dat, dat_meta, num_of_case)

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
            train_dat, dev_dat, test_dat, train_meta, dev_meta, test_meta = Utility.train_dev_test_split(dat, dat_meta)
        else:
            train_dat, test_dat, train_meta, test_meta = Utility.train_test_split(dat, dat_meta)

        print("\nPreparing train data...")
        # step 3: train data processing - feature selection - Anova and tukey
        train_dat_X = train_dat.iloc[:, :-2]
        train_dat_y = train_dat.iloc[:, -1]
        groups = train_dat.primary_site_code

        p_values = train_dat_X.apply(Utility.getANOVA, groups=groups, train_dat=train_dat)
        max_diffs = train_dat_X.apply(Utility.getMaxDiff, groups=groups)

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
        Utility.csv_to_tfrecords(trainfile)
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
            Utility.csv_to_tfrecords(devfile)
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
            Utility.csv_to_tfrecords(testfile)
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
            Utility.csv_to_tfrecords(testfile)
            testmetafile = outdir + "test1_meta.csv"
            test_meta.to_csv(testmetafile, index=False)
            print("\nAll done!")

    @staticmethod
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
            test.sort_index(axis=1, inplace=True)  # sort the column

        test['primary_code'] = test_label

        print("\nWriting to CSV...")
        filename1 = inputfile_test.split("/")[-1]
        test_file = outdir + filename1
        test.to_csv(test_file, index=False)
        print("\nConverting to tfrecords...")
        Utility.csv_to_tfrecords(test_file)

        print("\nWriting meta data to CSV...")
        test_meta = pd.read_csv(inputmetafile)
        test_meta = test_meta.loc[test_meta.primary_site.isin(labels),]

        filename2 = inputmetafile.split('/')[-1]
        testmeta_file = outdir + filename2
        test_meta.to_csv(testmeta_file, index=False)

    @staticmethod
    def test_prep_folder(testdat_dir, codesfile, feafile, outdir):
        '''
        GEO data preparation from different labs
        :param testdat_dir:
        :param codesfile:
        :param feafile:
        :param outdir:
        :return:
        '''
        files = os.listdir(testdat_dir)

        datfiles = [file for file in files if 'meta' not in file ]
        datfiles = [testdat_dir + file for file in datfiles]
        metafiles = [file for file in files if 'meta' in file]
        metafiles = [testdat_dir + file for file in metafiles]

        for file, metafile in zip(datfiles, metafiles):
            Data.test_prep(file, metafile, codesfile, feafile, outdir)

    @staticmethod
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

            Utility.csv_to_tfrecords(train_file)
            Utility.csv_to_tfrecords(test_file)
            fold += 1
