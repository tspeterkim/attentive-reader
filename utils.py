import numpy as np
import tensorflow as tf
import logging
from collections import Counter

def load_data(in_file, max_example=None):
    qst = []
    ans = []
    doc = []

    f = open(in_file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        q = line.strip().lower()
        a = f.readline().strip()
        d = f.readline().strip().lower()

        # TODO: what's the point of relabeling?

        qst.append(q)
        ans.append(a)
        doc.append(d)

        if max_example is not None and max_example <= len(ans):
            break

        f.readline()
    f.close()
    logging.info('# of Examples: %d' % len(ans))
    return doc, qst, ans


def build_dict(sentences, max_words=50000):
    wc = Counter()
    for s in sentences:
        for w in s.split(' '):
            wc[w] += 1

    ls = wc.most_common(max_words)
    logging.info('# of Words: %d -> %d' % (len(wc), len(ls)))

    for k in ls[:5]:
        logging.info(k)
    logging.info('...')
    for k in ls[-5:]:
        logging.info(k)

    return {w[0]: i+2 for (i,w) in enumerate(ls)}


def gen_embeddings(word_dict, dim, in_file=None):
    num_words = max(word_dict.values()) + 1
    embeddings = np.random.uniform(low=-0.01, high=0.01, size=(num_words, dim))
    logging.info('Embedding Matrix: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file at %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            v = line.split()
            assert len(v) == dim + 1
            if v[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[v[0]]:] = [float(x) for x in v[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' %
                        (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def get_minibatches(n, mb_size, shuffle=False):
    idx_list = np.arange(0, n, mb_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for i in idx_list:
        minibatches.append(np.arange(i, min(n, i+mb_size)))
    return minibatches


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(float)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def vectorize(examples, word_dict, entity_dict,
              sort_by_len=True, verbose=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(examples[0]), len(entity_dict))).astype(float)
    in_y = []
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        assert (a in d_words)
        seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words] # 0 for unk
        seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_l[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0
            in_y.append(entity_dict[a] if a in entity_dict else 0)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_l = in_l[sorted_index]
        in_y = [in_y[i] for i in sorted_index]

    return in_x1, in_x2, in_l, in_y
