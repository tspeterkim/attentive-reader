import sys
import time
import logging
import config
import utils

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
    word_dict = utils.build_dict(train_examples[0] + train_examples[2])
    entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + train_examples[1]))
    entity_markers = ['<entity_unk>'] + entity_markers
    entity_dict = {w : i for (i, w) in enumerate(entity_markers)}
    logging.info('# of Entity Markers: %d' % len(entity_dict))
    args.num_labels = len(entity_dict)

    logging.info('-' * 50)
    logging.info('Generating embedding...')
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    args.vocab_size, args.embedding_size = embeddings.shape


if __name__ == '__main__':
    args = config.get_args()

    args.train_file = 'data/cnn/train.txt'
    args.test_file = 'data/cnn/test.txt'
    args.dev_file = 'data/cnn/dev.txt'

    args.log_file = None
    args.debug = True

    args.embedding_file = 'data/glove.6B/glove.6B.50d.txt'
    args.embedding_size = utils.get_dim(args.embedding_file)

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    main(args)
