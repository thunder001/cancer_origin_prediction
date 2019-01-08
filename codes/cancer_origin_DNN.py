import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import time
import scipy as sp
import scipy.stats
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import tensorflow as tf
from shutil import copy
import os



def DNN_model(x, units=256):
    """
    Construct a DNN operation
    :param x: input feature tensor - [input_size]
    :param units: number of hidden layer units - [scalar]
    :return: prediction before activation - [number_of_class]
    """
    # nodes_input_layer = 10034
    # nodes_input_layer = 9601
    nodes_input_layer = 10360
    # nodes_input_layer = 10768
    # nodes_input_layer = 10737
    nodes_hidden_layer_1 = units
    nodes_hidden_layer_2 = units
    # nodes_hidden_layer_3 = units / 2

    tf.set_random_seed(seed=1)
    n_class = 18

    w1 = tf.get_variable("w1", [nodes_input_layer, nodes_hidden_layer_1],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.Variable(tf.zeros([nodes_hidden_layer_1]))
    l1 = tf.add(tf.matmul(x, w1), b1)
    l1 = tf.nn.relu(l1)

    w2 = tf.get_variable("w2", [nodes_hidden_layer_1, nodes_hidden_layer_2],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.Variable(tf.zeros([nodes_hidden_layer_2]))
    l2 = tf.add(tf.matmul(l1, w2), b2)
    l2 = tf.nn.relu(l2)

    # w3 = tf.get_variable("w3", [nodes_hidden_layer_2, nodes_hidden_layer_3],
    #                      initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # b3 = tf.Variable(tf.zeros([nodes_hidden_layer_3]))
    # l3 = tf.add(tf.matmul(l2, w3), b3)
    # l3 = tf.nn.relu(l3)

    out_w = tf.get_variable("out",[nodes_hidden_layer_2, n_class],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
    out_b = tf.Variable(tf.zeros([n_class]))
    out = tf.add(tf.matmul(l2, out_w), out_b)

    return out


def read_records(file_queue):
    """
    Read single example from file_quene

    :param file_queue: String - file quene
    :return: single example including features and label
    """

    reader = tf.TFRecordReader()
    key, record_string = reader.read(file_queue)
    features = tf.parse_single_example(
      record_string,
      features = {
          # 'features': tf.FixedLenFeature([10034], tf.float32),
          # 'features': tf.FixedLenFeature([9601], tf.float32),
          'features': tf.FixedLenFeature([10360], tf.float32),
          # 'features': tf.FixedLenFeature([10737], tf.float32),
          # 'features': tf.FixedLenFeature([10768], tf.float32),
          'label': tf.FixedLenFeature([], tf.int64),
      })

    example = tf.cast(features['features'], tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return example, label


def input_pipeline(filenames, batch_size, num_epochs):
    """
    Creade input pipeline from a list of input files

    :param filenames: input files
    :param batch_size:  int - size of batch
    :param num_epochs:  int - number of epochs
    :return: examples of batch size
    """

    filename_queue = tf.train.string_input_producer(
      [filenames], num_epochs=num_epochs, seed=0)
    example, label = read_records(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    # example_batch, label_batch = tf.train.shuffle_batch(
    #   [example, label], batch_size=batch_size, num_threads=1, capacity=capacity,
    #   min_after_dequeue=min_after_dequeue)

    example_batch, label_batch = tf.train.batch(
        [example, label], batch_size=batch_size, num_threads=1, capacity=capacity)
    return example_batch, label_batch


def input_pipeline_CV(filenames, batch_size, num_epochs):
    """
    Creadt input pipeline for a list of input files

    :param filenames: input files
    :param batch_size:  int - size of batch
    :param num_epochs:  int - number of epochs
    :return: examples of batch size
    """

    filename_queue = tf.train.string_input_producer(
      [filenames], num_epochs=num_epochs, seed=0)
    example, label = read_records(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, num_threads=1, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

    return example_batch, label_batch


def loss_compute(logits, labels):
    """
    Calculates the loss from the logits and the labels.

    :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    :param labels: Labels tensor, int32 - [batch_size].
    :return: loss: Loss tensor of type float.
    """

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')

    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """
    Sets the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    :param loss: Loss tensor, from loss().
    :param learning_rate: The learning rate to use for gradient descent.
    :return: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation_pr(logits, labels):
    """
    Compute average precision
    :param logits: logit tensor - [batch_size, number_of_class]
    :param labels: label tensor, int32 - [batch_size]
    :return: average_precision: dictionary of size of number of classes
    """

    pred_scores = tf.transpose(tf.nn.softmax(logits)).eval()
    true_labels = tf.transpose(tf.one_hot(labels, depth=18)).eval()
    precision, recall, _ = precision_recall_curve(true_labels.ravel(), pred_scores.ravel())
    average_precision = average_precision_score(true_labels, pred_scores, average="micro")
    return average_precision


def evaluation_acc(logits, labels):
    """
    Evaluate the overall accuracy of prediction
    :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    :param labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    :return: A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """

    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def compute_SSPN(conf_matrix_df):
    """ Compute both class wise SSPN and overall SSPN based on multiple class confusion matrix
    Note: SSPPA stands for specificity, sensitivity, positive predictive values, negative
    predictive value

    :param: conf_matrix_df: Pandas DataFrame - [number of prediction classes, number of prediction classes
    :return SSPN_df: Pandas DataFrame - [number of labelled classes, 5]
    """

    conf_matrix = conf_matrix_df.values
    print(conf_matrix)
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
            # acc = (TP + TN) / (TP + FP + TN + FN) // not necessary for multiple classes
            SSPN.append([sp, sn, ppv, npv])

    SSPN_array = np.array(SSPN)

    overall = np.mean(SSPN_array, axis=0)
    SSPN_index.append('Overall')

    SSPN_array = np.vstack((SSPN_array, overall))
    SSPN_df = pd.DataFrame(SSPN_array,
                            index=SSPN_index,
                            columns=['Specificity', 'Sensitivity', 'PPV', 'NPV'])

    return SSPN_df


def make_one_hot(labels, num_classes):
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


def compute_pr(logits, labels):
    """
    Compute class wise and overall precisions and recalls
    :param logits: softmax values of DNN model outputs
    :param labels: True sample labels
    :return: three dictionary objects that stores precisions, recalls
    and average precesions
    """

    precision = dict()
    recall = dict()
    average_precision = dict()

    labs_hot = make_one_hot(labels, 18)
    for i in range(labs_hot.shape[1]):
        # get precision recall and average precision for individual class
        if labs_hot[:, i].sum != 0:   # test if dataset contain this class
            precision[i], recall[i], _ = precision_recall_curve(labs_hot[:, i], logits[:, i])
            average_precision[i] = average_precision_score(labs_hot[:, i], logits[:, i], average="micro")

    # get overall precision recall and average precision
    precision[labs_hot.shape[1]], recall[labs_hot.shape[1]], _ = \
        precision_recall_curve(labs_hot.ravel(), logits.ravel())
    average_precision[labs_hot.shape[1]] = average_precision_score(labs_hot, logits, average="micro")

    return precision, recall, average_precision


def run_testing(testfile, testmetafile, units, modelfile, sample_size, codesfile):

    """ Run testing and return performance metrics

    :param testfile tfrecords formated testing file
    :param modelfile saved model for testing
    :param sample_size sample size for test data
    :param codesfile a csv file coding for cancer origins

    :return performance metrics, including precision, recall, average_precision,
    SSPN, comfusion matrix and overall accuracy
    """

    tf.reset_default_graph()

    features, labels = input_pipeline(testfile, batch_size=sample_size, num_epochs=1)
    logits = DNN_model(features, units=units)

    logits_soft = tf.nn.softmax(logits)   # shape - [sample_size, num_classes]
    pred_classes = tf.argmax(logits_soft, axis=1)    # shape - [sample_size, num_classes]
    labels_hot = tf.one_hot(labels, depth=18)   # shape - [sample_size, num_classes]

    accuracy = evaluation_acc(logits, labels)

    variable_to_restore = tf.trainable_variables()
    saver = tf.train.Saver(variable_to_restore)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        saver.restore(sess, modelfile)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        labs, logs_soft, preds, labs_hot, acc = None, None, None, None, None
        try:
            step = 1
            while not coord.should_stop():

                labs, logs_soft, preds, labs_hot, acc = \
                    sess.run([labels, logits_soft, pred_classes, labels_hot, accuracy])

                print("Iteration: {:d}".format(step))
                step += 1

        except tf.errors.OutOfRangeError:
            print('Testing done!\n')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)


        # 1. average_precision
        precision, recall, average_precision = compute_pr(logs_soft, labs)

        codes = pd.read_csv(codesfile, header=None)
        codes_map = dict(zip(codes[1], codes[0]))
        code_overall = len(codes_map)
        codes_map[code_overall] = "Overall"  # !Note: add overall precision to mapping dictionary

        average_precision_ser = pd.Series(average_precision)
        average_precision_ser.index = average_precision_ser.index.map(lambda x: codes_map[x])
        average_precision_ser.dropna(inplace=True)

        # 2. Multiple class comfusion matrix
        # Note: always square matrix, dimension is class number of true labels or predicted labels,
        # which is bigger
        cm = confusion_matrix(labs, preds)
        num_label_class = np.unique(labs).size
        num_pred_class = np.unique(preds).size

        test_class = []

        if num_label_class > num_pred_class:
            test_class = np.unique(labs).tolist()
        else:
            test_class = np.unique(preds).tolist()

        test_class.sort()
        test_index = [codes_map[x] for x in test_class]
        # print(test_index)
        cm_df = pd.DataFrame(cm, index=test_index, columns=test_index)

        # 3. SSPN matrix
        SSPN_df = compute_SSPN(cm_df)

        # 4. prediction result

        labs_name = list(map(lambda x: codes_map[x], labs))  # Note: map function returns a map object
        print(len(labs_name))
        preds_name = list(map(lambda y: codes_map[y], preds))
        print(len(preds_name))

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
            pred_df = pd.DataFrame(OrderedDict({'GEO_ID': geo_id,  'Primary_site': labs_name,
                                                'Prediction': preds_name, 'Correct': correct}))

        return precision, recall, average_precision_ser, SSPN_df, cm_df, acc, pred_df


def run_testing_CV(testfile, modelfile, units, sample_size, codesfile):

    """ Run testing and return performance metrics for cross-validation

    :param testfile tfrecords formated testing file
    :param modelfile saved model for testing
    :param sample_size sample size for test data
    :param codesfile a csv file coding for cancer origins

    :return performance metrics, including precision, recall, average_precision,
    SSPN, comfusion matrix and overall accuracy
    """

    tf.reset_default_graph()

    features, labels = input_pipeline_CV(testfile, batch_size=sample_size, num_epochs=1)
    logits = DNN_model(features, units)

    logits_soft = tf.nn.softmax(logits)   # shape - [sample_size, num_classes]
    pred_classes = tf.argmax(logits_soft, axis=1)    # shape - [sample_size, num_classes]
    labels_hot = tf.one_hot(labels, depth=18)   # shape - [sample_size, num_classes]

    accuracy = evaluation_acc(logits, labels)

    variable_to_restore = tf.trainable_variables()
    saver = tf.train.Saver(variable_to_restore)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        saver.restore(sess, modelfile)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        labs, logs_soft, preds, labs_hot, acc = None, None, None, None, None
        try:
            step = 1
            while not coord.should_stop():

                labs, logs_soft, preds, labs_hot, acc = \
                    sess.run([labels, logits_soft, pred_classes, labels_hot, accuracy])
                print(labs)
                print(logs_soft.shape)

                print("Iteration: {:d}".format(step))
                step += 1

        except tf.errors.OutOfRangeError:
            print('Testing done!\n')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)

        print(logs_soft)
        print(labs)
        # 1. average_precision
        precision, recall, average_precision = compute_pr(logs_soft, labs)

        codes = pd.read_csv(codesfile, header=None)
        codes_map = dict(zip(codes[1], codes[0]))
        code_overall = len(codes_map)
        codes_map[code_overall] = "Overall"  # !Note: add overall precision to mapping dictionary

        average_precision_ser = pd.Series(average_precision)
        average_precision_ser.index = average_precision_ser.index.map(lambda x: codes_map[x])
        average_precision_ser.dropna(inplace=True)

        # 2. Multiple class comfusion matrix
        # Note: always square matrix, dimension is class number of true labels or predicted labels,
        # which is bigger
        cm = confusion_matrix(labs, preds)
        num_label_class = np.unique(labs).size
        num_pred_class = np.unique(preds).size

        test_class = None
        if num_label_class > num_pred_class:
            test_class = np.unique(labs).tolist()
        else:
            test_class = np.unique(preds).tolist()

        test_class.sort()
        test_index = [codes_map[x] for x in test_class]
        # print(test_index)
        cm_df = pd.DataFrame(cm, index=test_index, columns=test_index)

        # 3. SSPN matrix
        SSPN_df = compute_SSPN(cm_df)

        return precision, recall, average_precision_ser, SSPN_df, cm_df, acc


def run_training(trainfiles, units, save_model_file):
    """Train cancer origin and save model into a file.

    :param trainfiles tfrecords formated file used for training model with specified hyperparameter,
    including batch_size, epoch number, learning rate and decay

    :param save_model_file model model file to be saved
    """

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        features, labels = input_pipeline(trainfiles, batch_size=128, num_epochs=30)

        logits = DNN_model(features, units=units)
        loss = loss_compute(logits, labels)

        # Add to the Graph operations that train the model.
        global_step = tf.Variable(0, trainable=False)

        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   1000, 0.96, staircase=True)

        train_op = training(loss, learning_rate)
        accuracy = evaluation_acc(logits, labels)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        saver = tf.train.Saver(tf.trainable_variables())

        # Create a session for running operations in the Graph.

        sess = tf.Session()

        # Initialize the variables (like the epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        feas, lals = sess.run([features, labels])
        print("Shape of features: {}".format(feas.shape))
        print("Shape of label: {}".format(lals.shape))
        # print("labels: {}".format(lals))

        # print("Trainable variables:")
        # for var in tf.trainable_variables():
        #     print(var)

        # print("Global variables:")
        # for var in tf.global_variables():
        #     print(var)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss])

                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    acc = sess.run(accuracy)
                    print('Step %d: loss = %.2f; accuracy = %.4f (%.3f sec)' % (step, loss_value, acc, duration))

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for 50 epochs, %d steps.' % step)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        saver.save(sess, save_path=save_model_file)

        sess.close()


def cross_validation(CVfiles_folder, CVModel_folder, units, folds, sample_size, codesfile):


    SSPN_overall = []

    for i in range(folds):
        print("\n\nRunning fold {0} of {1} ...".format(i, folds))
        trainfile = CVfiles_folder + "train_" + str(i) + ".tfrecords"
        print(trainfile)
        testfile = CVfiles_folder + "test_" + str(i) + ".tfrecords"
        print(testfile)
        CVModelfile = CVModel_folder + "model_" + str(i) + ".ckpt"
        print(CVModelfile)

        run_training(trainfile, units, CVModelfile)
        precision, recall, average_precision_ser, SSPN, cm, acc = \
            run_testing_CV(testfile, CVModelfile, units, sample_size, codesfile)

        overall = list(SSPN.loc["Overall"])
        overall.append(acc)
        # print("Overall {}".format(overall))
        SSPN_overall.append(overall)
        # print("SSPN Overall {}".format(SSPN_overall))

    SSPN_overall = pd.DataFrame(SSPN_overall, columns=['Specificity', 'Sensitivity',
                                             'Postive_predictive_value', 'Negative_predictive_value', 'Accuracy'])
    return SSPN_overall.transpose()


def run_model_selection(trainfile, testfile, testmetafile, model_dir, units, sample_size, codesfile, best_model_dir):

    # get models
    models = {}

    for i in range(len(units)):
        print("\n\nRunning training {0} of {1} ...".format(i + 1, len(units)))
        modelfile = model_dir + "model_" + str(i) + ".ckpt"
        print(units[i])
        print(type(units[i]))

        run_training(trainfile, units[i], modelfile)
        precision, recall, average_precision, SSPN, cm, acc, preds = \
            run_testing(testfile, testmetafile, units[i], modelfile,  sample_size, codesfile)

        models[units[i]] = acc
        # print("SSPN Overall {}".format(SSPN_overall))

    # choose best models (hidden layer unit)
    max_acc = 0
    for unit, acc in models.items():
        a = models[unit]
        if a > max_acc:
            max_acc = a
    print('Best accuracy is {}'.format(max_acc))
    for unit, acc in models.items():
        if acc == max_acc:
            best_unit = unit
            print('Best hidden units are {}'.format(best_unit))
            best_model_idx = units.index(best_unit)
            best_model_file1 = model_dir + "model_" + str(best_model_idx) + ".ckpt.index"
            best_model_file2 = model_dir + "model_" + str(best_model_idx) + ".ckpt.meta"
            best_model_file3 = model_dir + "model_" + str(best_model_idx) + ".ckpt.data-00000-of-00001"
            copy(best_model_file1, best_model_dir)
            copy(best_model_file2, best_model_dir)
            copy(best_model_file3, best_model_dir)


def mean_conf(SSPN, conf=0.95):
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


# #--------------------- Running in command line  -------------------
def main():
    args, _ = parser.parse_known_args()

    if args.run_type == "cv":

        CVData_dir = args.CVData_dir
        CVModel_dir = args.model_dir
        units = args.units[0]
        codesfile = args.codesfile
        n_fold = args.folds
        sample_size = args.sample_size
        results_dir = args.results_dir

        SSPN = cross_validation(CVData_dir, CVModel_dir, units, n_fold, sample_size, codesfile)
        print(SSPN)
        SSPN_stat = mean_conf(SSPN).round(4)
        print(SSPN_stat)
        SSPN_stat.to_csv(results_dir + 'sspn_stat.csv')

        # save metrics
        # spec_mean_ci = list(mean_confidence_interval(spec))
        # sens_mean_ci = list(mean_confidence_interval(sens))
        # ppn_mean_ci = list(mean_confidence_interval(ppn))
        # npn_mean_ci = list(mean_confidence_interval(npn))
        # accu_mean_ci = list(mean_confidence_interval(accu))
        #
        # sspn_raw = np.array([spec, sens, ppn, npn, accu])
        # np.savetxt("/Users/zhengc/Projects/cancer_origin/TCGA/cv_sspn_raw_results.csv", sspn_raw, delimiter=",")
        #
        # sspn_final = np.array([spec_mean_ci, sens_mean_ci,
        #                        ppn_mean_ci, npn_mean_ci, accu_mean_ci])
        # sspn_final = np.round(sspn_final, 4)
        # print("\nOverall model performace:")
        # pp.pprint(sspn_final)
        # np.savetxt("/Users/zhengc/Projects/cancer_origin/TCGA/cv_sspn_final_results.csv", sspn_final, delimiter=",")

    if args.run_type == "model_selection":
        trainfiles = args.trainfile
        testfile = args.testfile
        testmetafile = args.testmetafile
        model_dir = args.model_dir
        units = args.units
        sample_size = args.sample_size
        codesfile = args.codesfile
        best_model_dir = args.best_model_dir

        run_model_selection(trainfiles, testfile, testmetafile, model_dir, units, sample_size, codesfile, best_model_dir)

    if args.run_type == "train":
        trainfiles = args.trainfile
        units = args.units[0]
        modelfile = args.modelfile

        run_training(trainfiles, units, modelfile)

    if args.run_type == "test":
        testfile = args.testfile
        testmetafile = args.testmetafile
        units = args.units[0]
        modelfile = args.modelfile
        sample_size = args.sample_size
        codesfile = args.codesfile
        results_dir = args.results_dir

        precision, recall, average_precision, SSPN, cm, acc, preds = \
            run_testing(testfile, testmetafile, units, modelfile, sample_size, codesfile)


        # display metrics
        print("\nAccuracy: {}".format(round(acc,4)))
        print("\nConfusion matrix:")
        print(cm)
        print("\nAverage precision:")
        print(average_precision)
        print("\n SSPN:\n {}".format(SSPN))

        # save results to files

        pd.Series(acc, index=['accuracy']).to_csv(results_dir + "accuracy.txt", sep="\n")
        cm.to_csv(results_dir + "confusion_matrix.csv")
        SSPN.to_csv(results_dir + "sspn.csv")
        average_precision.to_csv(results_dir + "average_precision.csv")
        preds.to_csv(results_dir + "pred_results.csv", sep=",")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get performance of cancer origin prediction model using "
                                                 "test data")
    print(parser)

    parser.add_argument(
        'run_type',
        choices=["train", "test", "cv", "model_selection"],
        default='test',
        help='Choose the type of program to run'
    )

    parser.add_argument(
        '--trainfile',
        nargs="?",
        default="./Data/train_TCGA.tfrecords",
        help='Methylation file as tfrecords to be used for training model'
    )

    parser.add_argument(
        '--testfile',
        nargs="?",
        default="./Data/GEO.tfrecords",
        help='Methylation file to be tested as tfrecords format'
    )

    parser.add_argument(
        '--testmetafile',
        nargs="?",
        default="./Data/GEO_meta.csv",
        help='Meta data file to be used in testing'
    )

    parser.add_argument(
        '--modelfile',
        nargs="?",
        default="./Model/model_100.ckpt",
        help='Model file to be used in testing or to be saved in training.'
    )

    parser.add_argument(
        '--codesfile',
        nargs="?",
        default="./Data/code.csv",
        help='Map file from cancer origin name to numeric value as csv format.'
    )

    parser.add_argument(
        '--CVData_dir',
        nargs="?",
        default="./Data_test/CV_10/",
        help='Directory for storing methylation data for each fold'
    )
    parser.add_argument(
        '--model_dir',
        nargs="?",
        default="./Model/CV_model/",
        help='Directory for storing models for each fold'
    )

    parser.add_argument(
        '--units',
        nargs="*",
        default=[64, 128, 256],
        help='a list of hidden units to test'
    )

    parser.add_argument(
        '--best_model_dir',
        nargs="?",
        default="./Model/CV_model/",
        help='Directory for best model'
    )

    parser.add_argument(
        '--sample_size',
        nargs="?",
        type=int,
        choices=[1468, 701, 581, 448, 431, 143],
        help='Test sample size'
    )
    parser.set_defaults(sample_size=734)

    parser.add_argument(
        '-f', '--folds',
        type=int,
        default=10,
        help='Number of folds to be generated'
    )

    parser.add_argument(
        '--results_dir',
        nargs="?",
        default="./DNN_results/DNN_100_dev/",
        help='folder for test results'
    )

    main()


# trainfile ./DNN_data/DNN_100_data/train_dev_test/train1.tfrecords
# testfile ./DNN_data/DNN_100_data/train_dev_test/dev.tfrecords
# testmetafile ./DNN_data/DNN_100_data/train_dev_test/dev_meta.csv

# model_folder ./DNN_model/DNN_model_100_dev/
# units
# sample_size
# codesfile ./DNN_data/DNN_100_data/train_dev_test/code.csv
# best_model_dir ./DNN_model/DNN_model_100_dev/best_model/


# -------------------------Command line script---------------------------------------------

# model selection
# python3 ./python/cancer_origin_DNN_2.py model_selection \
#                                         --trainfile ./DNN_data/DNN_100_data/train_dev_test_15_10/train1.tfrecords \
#                                         --testfile ./DNN_data/DNN_100_data/train_dev_test_15_10/dev.tfrecords \
#                                         --testmetafile ./DNN_data/DNN_100_data/train_dev_test_15_10/dev_meta.csv  \
#                                         --model_dir ./DNN_model/DNN_model_100_dev_15_10/ \
#                                         --best_model_dir ./DNN_model/DNN_model_100_dev_15_10/best_model/ \
#                                         --codesfile ./DNN_data/DNN_100_data/train_dev_test_15_10/code.csv

# test set performance
# python3 ./python/cancer_origin_DNN_2.py test \
#                                         --testfile ./DNN_data/DNN_100_data/train_dev_test_15_10/test1.tfrecords \
#                                         --testmetafile ./DNN_data/DNN_100_data/train_dev_test_15_10/test1_meta.csv \
#                                         --units 128  \
#                                         --modelfile ./DNN_model/DNN_model_100_dev_15_10/best_model/model_1.ckpt \
#                                         --codesfile ./DNN_data/DNN_100_data/train_dev_test_15_10/code.csv \
#                                         --results_dir ./DNN_results/DNN_100_dev_80_10_10/test/

# metastatic cancer performance
# python3 ./python/cancer_origin_DNN_2.py test \
#                                         --testfile ./DNN_data/DNN_100_data/metastatic_data_15_20/metastatic_data1.tfrecords \
#                                         --testmetafile ./DNN_data/DNN_100_data/metastatic_data_15_20/metastatic_data1_meta.csv \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/DNN_model_100_dev_15_20/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/DNN_100_data/train_dev_test_15_20/code.csv \
#                                         --sample_size 701 \
#                                         --results_dir ./DNN_results/DNN_100_dev_60_20_20/metastatic/

# metastatic cancer performance 2
# python3 ./python/cancer_origin_DNN_2.py test \
#                                         --testfile ./DNN_data/metastatic_data2.tfrecords \
#                                         --testmetafile ./DNN_data/metastatic_data2_meta.csv \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/code.csv \
#                                         --sample_size 143 \
#                                         --results_dir ./DNN_results/metastatic2/

# dev set performance
# python3 ./python/cancer_origin_DNN_2.py test \
#                                         --testfile ./DNN_data/DNN_100_data/train_dev_test_15/dev.tfrecords \
#                                         --testmetafile ./DNN_data/DNN_100_data/train_dev_test_15/dev_meta.csv \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/DNN_model_100_dev_15/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/DNN_100_data/train_dev_test_15/code.csv \
#                                         --results_dir ./DNN_results/DNN_100_dev_60_20_20/dev/

# independent set performance
# python3 ./python/cancer_origin_DNN_2.py test \
#                                         --testfile ../GEO/test_data_100360/combined_final.tfrecords \
#                                         --testmetafile ../GEO/test_data_100360/combined_final_meta.csv \
#                                         --units 64  \
#                                         --modelfile ./DNN_model/DNN_model_100_dev_15/best_model/model_0.ckpt \
#                                         --codesfile ./DNN_data/DNN_100_data/train_dev_test_15/code.csv \
#                                         --sample_size 581 \
#                                         --results_dir ./DNN_results/DNN_100_dev_60_20_20/test_ind/

# cross validation performance
# python3 ./python/cancer_origin_DNN_2.py cv \
#                                         --CVData_dir ./DNN_data/DNN_100_data/cv_data_15_20/ \
#                                         --model_dir ./DNN_model/DNN_model_100_dev_15_20/cv_model/ \
#                                         --units 64  \
#                                         --codesfile ./DNN_data/DNN_100_data/train_dev_test_15_20/code.csv \
#                                         --sample_size 431 \
#                                         --results_dir ./DNN_results/DNN_100_dev_60_20_20/cv/



# CVData_dir = args.CVData_dir
#         CVModel_dir = args.model_dir
#         units = args.units
#         codesfile = args.codesfile
#         n_fold = args.folds
#         sample_size = args.sample_size
#         results_dir = args.results_dir
#
#         SSPN = cross_validation(CVData_dir, CVModel_dir, units, n_fold, sample_size, codesfile)
#         print(SSPN)