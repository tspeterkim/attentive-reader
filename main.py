import time
import logging
import config
import utils

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def gen_examples(x1, x2, l, y, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_l = l[minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y))
    return all_ex


def main(args):
    logging.info('-' * 50 + '')
    logging.info('Loading data...')
    if args.debug:
        train_examples = utils.load_data(args.train_file, 100)
        dev_examples = utils.load_data(args.dev_file, 100)
    else:
        train_examples = utils.load_data(args.train_file)
        dev_examples = utils.load_data(args.dev_file)

    args.num_train = len(train_examples[1])
    args.num_dev = len(dev_examples[1])

    logging.info('-' * 50)
    logging.info('Building dictionary...')
    word_dict = utils.build_dict(train_examples[0] + train_examples[1])
    entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')]
                                                            + train_examples[2]))
    entity_markers = ['<entity_unk>'] + entity_markers
    entity_dict = {w : i for (i, w) in enumerate(entity_markers)}
    logging.info('# of Entity Markers: %d' % len(entity_dict))
    args.num_labels = len(entity_dict)

    logging.info('-' * 50)
    logging.info('Generating embedding...')
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    embeddings = embeddings.astype('float32')
    args.vocab_size, args.embedding_size = embeddings.shape

    logging.info('-'* 50)
    logging.info('Creating TF computation graph...')
    # tf.reset_default_graph()
    d_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="d_input")
    q_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="q_input") # [batch_size, max_seq_length_for_batch]
    l_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name="l_mask") # [batch_size, entity num]
    y = tf.placeholder(dtype=tf.int32, shape=None, name="label") # batch size vector
    y_1hot= tf.placeholder(dtype=tf.float32, shape=(None, None), name="label_1hot") # onehot encoding of y [batch_size, entitydict]
    training = tf.placeholder(dtype=tf.bool)

    W_bilinear = tf.Variable(tf.random_normal([2*args.hidden_size, 2*args.hidden_size])) # TODO: random_normal?

    with tf.variable_scope('d_encoder'):
        d_embed = tf.nn.embedding_lookup(embeddings, d_input) # [batch, max_seq_length_for_batch, 50]
        d_embed_dropout = tf.layers.dropout(d_embed, rate=args.dropout_rate, training=training)
        if args.rnn_type == 'lstm':
            d_cell_fw = rnn.LSTMCell(args.hidden_size) # TODO: Dropout of 0.2
            d_cell_bw = rnn.LSTMCell(args.hidden_size)
        elif args.rnn_type == 'gru':
            d_cell_fw = rnn.GRUCell(args.hidden_size)
            d_cell_bw = rnn.GRUCell(args.hidden_size)

        d_outputs, _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, d_embed_dropout, dtype=tf.float32)
        d_output = tf.concat(d_outputs, axis=-1) # (batch, len, h)

    with tf.variable_scope('q_encoder'):
        q_embed = tf.nn.embedding_lookup(embeddings, q_input)
        d_embed_dropout = tf.layers.dropout(q_embed, rate=args.dropout_rate, training=training)
        if args.rnn_type == 'lstm':
            q_cell_fw = rnn.LSTMCell(args.hidden_size)
            q_cell_bw = rnn.LSTMCell(args.hidden_size)
        elif args.rnn_type == 'gru':
            q_cell_fw = rnn.GRUCell(args.hidden_size)
            q_cell_bw = rnn.GRUCell(args.hidden_size)
        q_outputs, q_laststates = tf.nn.bidirectional_dynamic_rnn(q_cell_fw, q_cell_bw, d_embed_dropout, dtype=tf.float32)
        if args.rnn_type == 'lstm':
            q_output = tf.concat([q_laststates[0][-1], q_laststates[1][-1]], axis=-1) # (batch, h)
        elif args.rnn_type == 'gru':
            q_output = tf.concat(q_laststates, axis=-1) # (batch, h)

    with tf.variable_scope('bilinear'):
        M = tf.expand_dims(tf.matmul(q_output, W_bilinear), axis=1) # [batch, h] -> [batch, 1, h]
        alpha = tf.nn.softmax(tf.reduce_sum(d_output * M, axis=2)) # [batch, len]
        bilinear_output = tf.reduce_sum(d_output * tf.expand_dims(alpha, axis=2), axis=1) # [batch]

    with tf.variable_scope('dense'):
        final_prob = tf.layers.dense(bilinear_output, units=args.num_labels, activation=tf.nn.softmax) # [batch, entity#]

    pred = final_prob * l_mask # ignore entities that don't appear in the passage
    train_pred = pred / tf.expand_dims(tf.reduce_sum(pred, axis=1), axis=1) # redistribute probabilities ignoring certain labels
    train_pred = tf.clip_by_value(train_pred, 1e-7, 1.0 - 1e-7) # TODO: why these values exactly?

    test_pred = tf.cast(tf.argmax(pred, axis=-1), tf.int32)
    acc = tf.reduce_sum(tf.cast(tf.equal(test_pred, y), tf.int32))

    loss_op = tf.reduce_mean(-tf.reduce_sum(y_1hot * tf.log(train_pred), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss_op)
    logging.info('Done!')

    logging.info('-'* 50)
    logging.info('Initial Test...')
    dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(dev_examples, word_dict, entity_dict)
    all_dev = gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    # TODO: first dev accuracy displays here
    dev_acc = 0.
    logging.info('Dev Accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    logging.info('-'*50)
    logging.info('Start training...')
    train_x1, train_x2, train_l, train_y = utils.vectorize(train_examples, word_dict, entity_dict)
    all_train = gen_examples(train_x1, train_x2, train_l, train_y, args.batch_size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start_time = time.time()
    n_updates = 0
    with tf.Session() as sess:
        sess.run(init)
        for e in range(args.num_epoches):
            np.random.shuffle(all_train)
            for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y) in enumerate(all_train):
                logging.info('Batch Size = %d, # of Examples = %d, max_len = %d' % (mb_x1.shape[0], len(mb_x1), mb_x1.shape[1]))

                y_label = np.zeros((mb_x1.shape[0], args.num_labels))
                for r, i in enumerate(mb_y): # convert (batch) -> (batch, entity_size)
                    y_label[r][i] = 1.

                train_loss = sess.run(loss_op, feed_dict={d_input:mb_x1, q_input:mb_x2, y_1hot: y_label, l_mask: mb_l, training: True})
                logging.info('Epoch = %d, Iter = %d (max = %d), Loss = %.2f, Elapsed Time = %.2f (s)' %
                                (e, idx, len(all_train), train_loss, time.time() - start_time))
                n_updates += 1

                if n_updates % args.eval_iter == 0:
                    saver.save(sess, args.model_path, global_step=e)
                    correct = 0
                    n_examples = 0
                    for d_x1, d_mask1, d_x2, d_mask2, d_l, d_y in all_dev:
                        correct += sess.run(acc, feed_dict = {d_input:mb_x1, q_input:mb_x2, y: mb_y, l_mask: mb_l, training: False})
                        n_examples += len(d_x1)
                    dev_acc = correct * 100. / n_examples
                    logging.info('Dev Accuracy: %.2f %%' % dev_acc)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        logging.info('Best Dev Accuracy: epoch = %d, n_updates (iter) = %d, acc = %.2f %%' %
                                        (e, n_updates, dev_acc))

        logging.info('-'*50)
        logging.info('Training Finished...')
        logging.info("Model saved in file: %s" % saver.save(sess, args.model_path))



if __name__ == '__main__':
    args = config.get_args()

    # TODO: move this to config.py
    args.train_file = 'data/cnn/train.txt'
    args.test_file = 'data/cnn/test.txt'
    args.dev_file = 'data/cnn/dev.txt'

    args.log_file = 'log/log.txt' # if not specifed, prints all info to console
    # args.log_file = None
    args.debug = True # if true, use only the first 100 training/dev examples

    args.embedding_file = 'data/glove.6B/glove.6B.50d.txt'
    args.embedding_size = utils.get_dim(args.embedding_file)

    args.model_path = "model/attreader"

    args.batch_size = 32
    args.num_epoches = 100
    args.eval_iter = 100
    args.hidden_size = 128
    args.num_layers = 1
    args.bidir = True
    args.att_func = 'bilinear'
    args.grad_clipping = 10.0
    args.optimizer = 'sgd'
    args.learning_rate = 0.1
    args.dropout_rate = 0.2
    args.rnn_type = 'gru' # or 'lstm'

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    main(args)
