import os

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison


class Utility(object):

    # def _int64_feature(value):
    #     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    #
    # def _bytes_feature(value):
    #     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #
    # def _float_feature(value):
    #     return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def train_dev_test_split(df, df_meta, train_frac=0.60, dev_frac=0.2, seed=1):
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        df_meta_sub = df_meta.loc[df.primary_site.isin(samples.index),]
        return df_sub, df_meta_sub


# dat_dir = '/Users/zhengc/Projects/cancer_origin_2/Cancer_origin_prediction/data/CV/'
# files = os.listdir(dat_dir)
# for file in files:
#     datfile = dat_dir + file
#     Utility.csv_to_tfrecords(datfile)


