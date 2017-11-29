import argparse

def str2bool(v):
    return v.lower() in ['yes','true','1','t','y']

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool) # TODO: what does this exactly do

    parser.add_argument('--debug',
                        type='bool',
                        default=True,
                        help='whether it is debug mode i.e. use only first 100 examples')

    parser.add_argument('--test_only',
                        type='bool',
                        default=False,
                        help='test_only: no need to run training process')

    parser.add_argument('--train_file',
                        type=str,
                        default='data/cnn/train.txt',
                        help='Training file')

    parser.add_argument('--dev_file',
                        type=str,
                        default='data/cnn/dev.txt',
                        help='Development file')

    parser.add_argument('--test_file',
                        type=str,
                        default='data/cnn/test.txt',
                        help='Test file')

    parser.add_argument('--model_path',
                        type=str,
                        default='model/attreader.ckpt',
                        help='Model path to save TF checkpoints')

    parser.add_argument('--log_file',
                        type=str,
                        default='log/log.txt',
                        help='Log file')

    parser.add_argument('--embedding_file',
                        type=str,
                        default='data/glove.6B/glove.6B.50d.txt',
                        help='Word embedding file')

    parser.add_argument('--embedding_size',
                        type=int,
                        default=None,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')

    parser.add_argument('--bidir',
                        type='bool',
                        default=True,
                        help='bidir: whether to use a bidirectional RNN')

    parser.add_argument('--rnn_type',
                        type=str,
                        default='gru',
                        help='RNN type: lstm or gru (default)')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')

    parser.add_argument('--num_epoches',
                        type=int,
                        default=100,
                        help='Number of epoches')

    parser.add_argument('--eval_iter',
                        type=int,
                        default=100,
                        help='Evaluation on dev set after K updates')

    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.2,
                        help='Dropout rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help='Optimizer: sgd (default) or adam or rmsprop')

    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate for SGD')

    parser.add_argument('--grad_clipping',
                        type=float,
                        default=10.0,
                        help='Gradient clipping')





    return parser.parse_args()
