import tensorflow as tf
import numpy as np
import math
import os
import collections

"""
def read_dataset(file_path):
    corpus = []
    with open(file_path) as f:
        for line in f.readlines():
            words = line.strip().split()
            corpus.extend(words)

    return corpus
"""

file_path = "test_small.corpus"
#corpus = read_dataset(file_path)

#print(corpus)

batch_size = 5
num_of_iters = 10


vocabulary_size = 10
embedding_size = 128
num_of_neg_samples = 10

log_dir = "./tf_test.log"

if not os.path.exists(log_dir):
  os.makedirs(log_dir)


"""
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

"""

def read_dataset(dataset_path):

    corpus = []
    with open(dataset_path, 'r') as f:
        for doc in  f.readlines():
            words = doc.strip().split()
            corpus.append(words)

    return corpus

corpus = read_dataset(file_path)


def build_dataset(corpus, vocabulary_size):
    word_count_pairs = [['UNK', -1]] # unknown word
    word_count_pairs.extend(collections.Counter([word for line in corpus for word in line]).most_common(vocabulary_size - 1)) # get the most common words

    word2id = dict()
    for word, _ in word_count_pairs:
        word2id[word] = len(word2id)  # label each word with a number

    data = list()
    unk_count = 0
    for doc in corpus:
        data.append([])
        for word in doc:
            index = word2id.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            data[-1].append(index)

    word_count_pairs[0][1] = unk_count
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return data, word_count_pairs, word2id, id2word


def generate_word_context_pairs(batch_size, windows_size, ):


"""

def generate_batch(data, batch_size, window_size):
    global data_index

    assert batch_size % window_size == 0

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span_size = 2*window_size + 1

    buffer = collections.deque(maxlen=span_size)

    if data_index + span_size > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span_size])
    data_index += span_size

    for i in range(batch_size / window_size):
        context_words = [for w in range(span_size) if w != window_size]
        for j, word in enumerate(context_words):
            batch[i*window_size + j] = buffer[window_size] # base
            labels[i*window_size + j, 0] = buffer[word]

        if data_index == len(data):
            buffer.extend(data[0:span_size])
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span_size) % len(data)

    return batch_inputs, batch_labels

generate_batch(data=corpus, batch_size=10, window_size=5)


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

    # Add the loss to the summary
    tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)


    # Get the emebddings
    #norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    #normalized_embeddings = embeddings / norm
    normalized_embeddings = embeddings


    # Merge all summaries.
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:

    summary_file_writer = tf.summary.FileWriter(log_dir, session.graph)

    init.run()
    print("initialized!")

    average_loss = 0.0
    for iter in xrange(num_of_iters):


        batch_inputs, batch_labels = generate_batch(batch_size, )
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, summary_value, loss_value = session.run([optimizer, merged_summary, loss], feed_dict=feed_dict)
        average_loss += loss_value

        summary_file_writer.add_summary(summary_value, iter)

        if iter % 2000 == 0:
            if iter > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss over the last 2000 batches: {}".format(average_loss))
            average_loss = 0.0

    final_embeddings = normalized_embeddings.eval()


summary_file_writer.close()
"""

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