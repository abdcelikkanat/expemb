import tensorflow as tf
import numpy as np
import math

vocab_size = 10
batch_size = 15
embed_size = 4

w0 = tf.Variable(tf.truncated_normal([vocab_size, embed_size],
                                     stddev=1.0 / math.sqrt(embed_size)))

w1 = tf.Variable(tf.truncated_normal([vocab_size, embed_size],
                                     stddev=1.0 / math.sqrt(embed_size)))

train_input = tf.cast(tf.constant(np.random.choice(range(vocab_size), batch_size), shape=[batch_size, 1]), dtype=tf.int64)
train_label = tf.cast(tf.constant(np.random.choice(range(vocab_size), batch_size), shape=[batch_size, 1]), dtype=tf.int64)

neg_samp_inx = tf.nn.fixed_unigram_candidate_sampler(
    true_classes=train_label,
    num_true=1,
    num_sampled=3*batch_size,
    unique=False,
    range_max=10,
    distortion=0.75,
    unigrams=[1 for _ in range(vocab_size)]
)[0]

cv0 = tf.squeeze(tf.gather(w0, train_input))
cv1 = tf.squeeze(tf.gather(w1, train_label))

mm = tf.reduce_sum(tf.multiply(cv0, cv1), axis=1)
loss = tf.sigmoid(mm)

cv0_neg = tf.tile(cv0, [3,1])
cv1_neg = tf.squeeze(tf.gather(w1, neg_samp_inx))
mm_neg = tf.multiply(cv0, cv1)
loss_neg = tf.reduce_sum(tf.sigmoid(-mm_neg), axis=1)

loss = +tf.reduce_sum(loss) + tf.reduce_sum(loss_neg)


sess = tf.Session()

sess.run(tf.global_variables_initializer())
print(sess.run(loss))
