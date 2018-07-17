import tensorflow as tf
import numpy as np
import collections
import math
import numpy as np






log_dir = "./tf.log"
num_of_iters = 10000
batch_size = 10000
model_dim = 4
input_size = 10

#print(len(target_list))
# Tensorflow

graph = tf.Graph()

def generate_dataset():

    p = 0.3
    N = batch_size

    x = np.random.choice([0, 1], N, [1.0-p, p])

    return x


x = generate_dataset()


with graph.as_default():
    with tf.name_scope('inputs'):
        # X  = tf.placeholder(tf.int32, shape=[batch_size])
        train_inputs = tf.placeholder(tf.float32, shape=[batch_size])

    # Define variables, Construct the variables for the NCE loss
    with tf.name_scope('eta'):
        eta = tf.Variable(
            tf.truncated_normal(
                [model_dim, 1],
                stddev=1.0 / math.sqrt(model_dim)))

    # Define variables, Construct the variables for the NCE loss
    with tf.name_scope('T_weights'):
        T_weights = tf.Variable(
            tf.truncated_normal(
                [input_size, model_dim],
                stddev=1.0 / math.sqrt(model_dim)))
    with tf.name_scope('T_biases'):
        T_biases = tf.Variable(tf.zeros([input_size, 1]))


    T_val = tf.matmul(T_weights, train_inputs)

    teta = tf.add(T_val, T_biases)

    tetamul = tf.matmul(T_weights, eta)



    loss = teta

    # Add the loss to the summary
    tf.summary.tensor_summary('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(-loss)


    # Merge all summaries.
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

doc_index = 0
with tf.Session(graph=graph) as session:

    summary_file_writer = tf.summary.FileWriter(log_dir, session.graph)

    init.run()
    print("initialized!")

    average_loss = 0.0
    for iter in xrange(num_of_iters):

        feed_dict = {train_inputs: x}

        _, summary_value, loss_value = session.run([optimizer, merged_summary, loss], feed_dict=feed_dict)
        average_loss += loss_value

        summary_file_writer.add_summary(summary_value, iter)

        if iter % 2000 == 0:
            if iter > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss over the last 2000 batches: {}".format(average_loss))
            average_loss = 0.0

    final_eta = eta.eval()


summary_file_writer.close()


p = 1.0 / (1.0 + np.exp(-final_eta))

print("P value: {}".format(p))