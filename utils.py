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
    return qst, ans, doc


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
