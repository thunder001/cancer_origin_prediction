import tensorflow as tf
import time
from DNN_model import DNN_model
from input_pipeline import InputPipeline


class ModelTraining(object):
    """
    Train the model
    """

    def _loss_compute(self, logits, labels):
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

    def _training(self, loss, learning_rate):
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

    def run_training(self, trainfiles, units, save_model_file):
        """Train cancer origin and save model into a file.

        :param trainfiles tfrecords formated file used for training model with specified hyperparameter,
        including batch_size, epoch number, learning rate and decay
        :units number of hidden layer units
        :param save_model_file model model file to be saved
        """

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            ip = InputPipeline()
            features, labels = ip.input_pipeline(trainfiles, batch_size=128, num_epochs=30)

            logits = DNN_model(features, units=units)
            loss = self._loss_compute(logits, labels)

            # Add to the Graph operations that train the model.
            global_step = tf.Variable(0, trainable=False)

            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)

            train_op = self._training(loss, learning_rate)
            # accuracy = evaluation_acc(logits, labels)
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
                print('Done training for 30 epochs, %d steps.' % step)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

            saver.save(sess, save_path=save_model_file)

            sess.close()
