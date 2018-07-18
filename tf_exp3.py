import tensorflow as tf
import numpy as np
import collections
import math


file_path = "citeseer_n80_l10_w10_k80_deepwalk_node_corpus.corpus"


def read_dataset(file_path):

    corpus = []
    with open(file_path, 'r') as f:
        for doc in f.readlines():
            words = doc.strip().split()
            corpus.append(words)

    return corpus


def build_vocab(corpus, vocabulary_size=None):
    if vocabulary_size is None:
        num_of_words = None
    else:
        num_of_words = vocabulary_size - 1

    word_count_pairs = [['UNK', -1]]  # unknown word
    word_count_pairs.extend(collections.Counter([word for line in corpus for word in line]).most_common(num_of_words))  # get the most common words

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


def get_batch(doc_index, window_size, data):
    # it is assumed that each batch size is the length of a document
    L = len(data[doc_index])  # Length of the document

    doc = data[doc_index]

    batch_size = 3*(window_size**2) - window_size + ( len(corpus[0]) - (2*window_size) ) * (2*window_size)
    target_list = np.ndarray(shape=(batch_size), dtype=np.int32)
    context_list = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    #print(doc)
    inx = 0

    for target_inx, target_word in enumerate(doc):
        context_max_inx = min(target_inx+window_size+1, L)
        context_min_inx = max(target_inx-window_size, 0)
        context = [doc[j] for j in range(context_min_inx, context_max_inx) if j != target_inx]
        for context_word in context:
            #target_list.append(target_word)
            #context_list.append([context_word, 0])
            target_list[inx] = target_word
            context_list[inx, 0] = context_word
            inx += 1

    return target_list, context_list



corpus = read_dataset(file_path)
data, word_count_pairs, word2id, id2word = build_vocab(corpus)
#target_list, context_list = get_batch(doc_index=0, window_size=5, data=data)


vocabulary_size = len(word_count_pairs)
embedding_size = 128

log_dir = "./tf.log"
num_of_iters = 150000
window_size = 10
batch_size = 3*(window_size**2) - window_size +  ( len(corpus[0]) - (2*window_size) ) * (2*window_size)
num_of_neg_samples = 5


#print(len(target_list))
# Tensorflow

graph = tf.Graph()

def nce_loss(true_logits, sampled_logits, batch_size):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / batch_size
    return nce_loss_tensor

with graph.as_default():
    with tf.name_scope('inputs'):
        # X  = tf.placeholder(tf.int32, shape=[batch_size])
        train_inputs = tf.placeholder(tf.int32, shape=[1])
        train_labels = tf.placeholder(tf.int32, shape=[1, 1])

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

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([vocabulary_size, embedding_size]),
            name="sm_w_t")


    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(train_labels, dtype=tf.int64), [1, 1])

    unigrams = [0 for _ in range(vocabulary_size)]
    #for pair in word_count_pairs:
    #    unigrams[word2id[pair[0]]] = pair[1]
    for i in range(len(unigrams)):
        unigrams[i] = word_count_pairs[i][1]

    """
    neg_samp_inx = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=num_of_neg_samples * 1,
        unique=False,
        range_max=vocabulary_size,
        distortion=0.75,
        unigrams=unigrams
    )[0]

    cv0 = tf.squeeze(tf.gather(embeddings, train_inputs))
    cv1 = tf.squeeze(tf.gather(nce_weights, train_labels))

    mm = tf.reduce_sum(tf.multiply(cv0, cv1), axis=1)
    loss = -tf.log_sigmoid(mm)

    cv0_neg = tf.tile(cv0, [num_of_neg_samples, 1])
    cv1_neg = tf.squeeze(tf.gather(nce_weights, neg_samp_inx))
    mm_neg = tf.multiply(cv0_neg, cv1_neg)
    loss_neg = -tf.log_sigmoid(-mm_neg)

    loss = tf.reduce_sum(loss) + tf.reduce_sum(loss_neg)
    loss /= ((batch_size+1)*num_of_neg_samples)
    """

    loss = tf.reduce_sum(tf.nn.nce_loss(weights=nce_weights,
                          biases=nce_biases,
                          labels=train_labels,
                          inputs=embed,
                          num_sampled=5,
                          num_classes=vocabulary_size,
                          sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                           true_classes=labels_matrix,
                           num_true=1,
                           num_sampled=num_of_neg_samples,
                           unique=False,
                           range_max=vocabulary_size,
                           distortion=0.75,
                           unigrams=unigrams)  )
                         )


    # Add the loss to the summary
    tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)


    # Get the emebddings
    #norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    #normalized_embeddings = embeddings / norm
    normalized_embeddings = nce_weights


    # Merge all summaries.
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

doc_index = 0
word_pos = 0
with tf.Session(graph=graph) as session:

    summary_file_writer = tf.summary.FileWriter(log_dir, session.graph)

    init.run()
    print("initialized!")

    average_loss = 0.0
    for iter in xrange(num_of_iters):

        batch_inputs, batch_labels = get_batch(doc_index, window_size, data)

        feed_dict = {train_inputs: [batch_inputs[word_pos]], train_labels: [[batch_labels[word_pos, 0]]]}

        word_pos += 1
        if word_pos >= len(data[0]):
            word_pos = 0
            doc_index += 1
        if doc_index >= len(data):
            doc_index = 0


        _, summary_value, loss_value = session.run([optimizer, merged_summary, loss], feed_dict=feed_dict)
        average_loss += loss_value

        summary_file_writer.add_summary(summary_value, iter)

        if iter % 2000 == 0:

            if iter > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("{} Average loss over the last 2000 batches: {}".format(iter, average_loss))
            average_loss = 0.0

    final_embeddings = normalized_embeddings.eval()

summary_file_writer.close()

#print([len(final_embeddings)])
embed_file = "output4.embedding"
with open(embed_file, 'w') as f:
    f.write("{} {}\n".format(vocabulary_size-1, embedding_size))
    inx_list = range(len(final_embeddings))
    #np.random.shuffle(inx_list)
    for i in inx_list:
        if id2word[i] != 'UNK':
            f.write("{} {}\n".format(id2word[i], " ".join(str(val) for val in final_embeddings[i])))


