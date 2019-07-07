
import tensorflow as tf


class InputPipeline(object):

    def _read_records(self, file_queue):
        """
        Read single example from file_quene

        :param file_queue: String - file quene
        :return: single sample including features and label
        """

        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_queue)
        features = tf.parse_single_example(
            record_string,
            features={
                'features': tf.FixedLenFeature([10360], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        sample = tf.cast(features['features'], tf.float32)
        label = tf.cast(features['label'], tf.int64)
        return sample, label

    def input_pipeline(self, filenames, batch_size, num_epochs):
        """
        Creade input pipeline from a list of input files

        :param filenames: input files
        :param batch_size:  int - size of batch
        :param num_epochs:  int - number of epochs
        :return: samples of batch size
        """

        filename_queue = tf.train.string_input_producer(
            [filenames], num_epochs=num_epochs, seed=0)
        sample, label = self._read_records(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        # example_batch, label_batch = tf.train.shuffle_batch(
        #   [example, label], batch_size=batch_size, num_threads=1, capacity=capacity,
        #   min_after_dequeue=min_after_dequeue)

        sample_batch, label_batch = tf.train.batch(
            [sample, label], batch_size=batch_size, num_threads=1, capacity=capacity)
        return sample_batch, label_batch
