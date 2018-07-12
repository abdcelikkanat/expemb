import tensorflow as tf
import numpy as np
import math


def read_dataset(file_path):
    corpus = []
    with open(file_path) as f:
        for line in f.readlines():
            words = line.strip().split()
            corpus.extend(words)

    return corpus


file_path = "test_small.corpus"
corpus = read_dataset(file_path)

print(corpus)

batch_size = 5
num_of_iters = 10


vocabulary_size = 10
embedding_size = 128
num_of_neg_samples = 10

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('inputs'):
        # X  = tf.placeholder(tf.int32, shape=[batch_size])
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # Get the embeddings
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Define variables, Construct the variables for the NCE loss
    with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_of_neg_samples,
                                             num_classes=vocabulary_size))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:

    init.run()
    print("initialized!")

    average_loss = 0.0
    for iter in xrange(num_of_iters):
        feed_dict = {train_inputs: corpus[0:5], train_labels: }
        _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict)
        average_loss += loss_val


    for val in feed_dict:
        print(val)

        s = session.run(y, feed_dict=feed_dict)
        print(s)


"""
graph = tf.Graph()

with graph.as_default():
    
    with tf.name_scope('inputs'):
        #X  = tf.placeholder(tf.int32, shape=[batch_size])
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        y = tf.square(train_inputs)
        train_input_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    init = tf.global_variables_initializer()


with tf.Session(graph=graph) as session:
    init.run()

    feed_dict = {train_inputs: corpus[0:5]}

    for val in feed_dict:
        print(val)

        s = session.run(y, feed_dict=feed_dict)
        print(s)

"""