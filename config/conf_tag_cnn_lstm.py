import argparse

def init_arguments(parser):
    # MODEL
    parser.add_argument('--model_path', type=str, default='models.clf.cnn_lstm', help='model_path')
    parser.add_argument('--model_name', type=str, default='CNN_LSTM', help='model_name')
    parser.add_argument('--rnn_type', type=str, default='GRU', help='type of RNN')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cnn_filter_size', type=int, default=3, help='filter size of cnn')
    parser.add_argument('--cnn_encoding_dim', type=int, default=100, help='dimionsions of cnn output')

    # TRAINING
    parser.add_argument('--max_epoch', type=int, default=400, help='max number of epoches')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='grad_clip')
    parser.add_argument('--learning-_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay rate')
    parser.add_argument('--decay_steps', type=int, default=100, help='decay steps')
    parser.add_argument('--keep_rate', type=float, default=0.5, help='keep_rate')
    parser.add_argument('--show_every', type=int, default=100, help='number of batches between showing results')
    parser.add_argument('--save_every', type=int, default=2000, help='number of batches between saveing the model')
    parser.add_argument('--valid_every', type=int, default=400, help='number of batches between validating the model')
    parser.add_argument('--device', type=str, default='/cpu:0', help='computing device')

    # DATASET
    parser.add_argument('--data_shape', type=list, default=[64,64,3,100], help='data shape')
    parser.add_argument('--label_shape', type=list, default=[100], help='data shape')

    # ENVIRONMENT
    parser.add_argument('--session_name', type=str, default='test', help='session name')
