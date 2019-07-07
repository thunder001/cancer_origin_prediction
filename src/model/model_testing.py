import tensorflow as tf
from DNN_model import DNN_model
from input_pipeline import InputPipeline


def run_testing(testfile, modelfile, units, sample_size):

    """ Run testing

    :param testfile tfrecords formated testing file
    :param modelfile saved model for testing
    :units numer of neurons of hidden layers
    :param sample_size sample size for test data

    :return performance metrics, including precision, recall, average_precision,
    SSPN, comfusion matrix and overall accuracy
    """

    tf.reset_default_graph()
    ip = InputPipeline()
    features, labels = ip.input_pipeline(testfile, batch_size=sample_size, num_epochs=1)
    logits = DNN_model(features, units=units)

    logits_soft = tf.nn.softmax(logits)   # shape - [sample_size, num_classes]
    pred_classes = tf.argmax(logits_soft, axis=1)    # shape - [sample_size, num_classes]
    labels_hot = tf.one_hot(labels, depth=18)   # shape - [sample_size, num_classes]

    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    variable_to_restore = tf.trainable_variables()
    saver = tf.train.Saver(variable_to_restore)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)
        print(testfile)
        print(modelfile)

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
            print('Testing done!')
            print('Accuracy: {}'.format(acc))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)

    return labs, logs_soft, preds, labs_hot, acc

