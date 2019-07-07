
import tensorflow as tf


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

    out_w = tf.get_variable("out", [nodes_hidden_layer_2, n_class],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
    out_b = tf.Variable(tf.zeros([n_class]))
    out = tf.add(tf.matmul(l2, out_w), out_b)

    return out
