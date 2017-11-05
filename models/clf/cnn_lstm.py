import tensorflow as tf
import logging


class CNNLSTM():
    def __init__(self, args):
        self._logger = logging.getLogger(__name__)
        self.args = args

    def _create_placeholders(self):
        args = self.args
        data_shape = [None]
        data_shape.extend(args.data_shape)
        label_shape = [None]
        label_shape.extend(args.label_shape)

        self.input_plh = tf.placeholder(
                dtype=tf.float32,
                shape=data_shape,
                name='input_plh')

        self.label_plh = tf.placeholder(
                dtype=tf.int32,
                shape=label_shape,
                name='label_plh')

        self.is_training = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name='is_training_plh')
        self.keep_prob = tf.placeholder(
                dtype=tf.float32,
                shape=[],
                name='keep_prob')

    def init_global_step(self):
        args = self.args
        # Global steps for asynchronous distributed training.
        with tf.device(args.device):
            self.global_step = tf.get_variable('global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)

    def _conv2d(input_, k_h, k_w, o_c, scope='conv2d', padding='SAME'):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], o_c])
            b = tf.get_variable('b', [o_c])
            return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding=padding) + b

    def _max_pool(input_, k_h, k_w):
        return tf.nn.max_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='SAME')

    def _cnn(self, cnn_input, scope='CNN'):
        """
        : cnn_input         input float tensor of shape [BATCH_SIZE * MAX_LSTM_STEP, DATA_HEIGHT, DATA_HEIGHT, channel_num]
        : cnn_output        output float tensor of shape [BATCH_SIZE * MAX_LSTM_STEP, cnn_encoding_dim]
        """
        args = self.args
        k_h = args.cnn_filter_size
        k_w = args.cnn_filter_size
        features = [16, 32, 64, 128, 256]
        fc6_features = 256

        with tf.variable_scope(scope):
            conv1 = self._conv2d(cnn_input, k_h, k_w, features[0], scope='conv1')
            relu1 = tf.nn.relu(conv1)
            pool1 = self._max_pool(relu1, 2, 2)

            conv2 = self._conv2d(pool1, k_h, k_w, features[1], scope='conv2')
            relu2 = tf.nn.relu(conv2)
            pool2 = self._max_pool(relu2, 2, 2)

            conv3 = self._conv2d(pool2, k_h, k_w, features[2], scope='conv3')
            relu3 = tf.nn.relu(conv3)
            pool3 = self._max_pool(relu3, 2, 2)

            conv4 = self._conv2d(pool3, k_h, k_w, features[3], scope='conv4')
            relu4 = tf.nn.relu(conv4)
            pool4 = self._max_pool(relu4, 2, 2)

            conv5 = self._conv2d(pool4, k_h, k_w, features[4], scope='conv5')
            relu5 = tf.nn.relu(conv5)
            pool5 = self._max_pool(relu5, 2, 2)

            """
            conv6 = self._conv2d(pool5, k_h, k_w, features[5], scope='conv6')
            relu6 = tf.nn.relu(conv6)
            pool6 = self._max_pool(relu6, 2, 2)
            """

            fc6   = self._conv2d(pool5, 2, 2, fc6_features, padding='VALID', scope='fc6')
            relu6 = tf.nn.relu(fc6)
            dropout = tf.nn.dropout(relu6, self.keep_prob)

            fc7   = self._conv2d(dropout, 1, 1, args.cnn_encoding_dim, padding='VALID', scope='fc7')
            fc7   = tf.squeeze(fc7, [1, 2])
            return fc7

    def _bilstm(self, bilstm_input, scope='biLSTM'):
        shape = bilstm_input.get_shape()
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [shape[1], args.num_classes])
            b = tf.get_variable('b', [args.num_classes])
            return tf.matmul(bilstm_input, w) + b

    def _inference_graph(self):
        """
        :self.input_plh     input float tensor of shape [BATCH_SIZE, DATA_HEIGHT, DATA_HEIGHT, channel_num, MAX_LSTM_STEP]
        :logits             output float tensor of shape [BATCH_SIZE, num_classes]
        """

        """ Apply convolutions """
        args = self.args
        # cnn_output has a shape: [BATCH_SIZE, cnn_encoding_dim, MAX_LSTM_STEP]
        cnn_input = tf.transpose(self.input_plh, [0, 4, 1, 2, 3])
        shape = tf.shape(cnn_input)
        cnn_input = tf.reshape(input_plh, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

        cnn_output = self._cnn(self.input_plh, scope='CNN')
        # cnn_output = tf.reshape(cnn_output, [shape[0], shape[1], -1])
        # cnn_output = tf.transpose(cnn_output, [0, 2, 1])

        """ Apply LSTM """
        self.logits = self._bilstm(cnn_output, scope='biLSTM')

    def async_training_op(self, cost, embd_var, other_var_list,
            grad_clip=5.0,
            max_norm=200.0,
            learning_rate=0.01,
            grads=None,
            train_embd=True):
        '''
        2016-11-15, Haoze Sun
        0. gradient for word embeddings is a tf.sparse_tensor
        1. clip_norm and clip by value operations do not support sparse tensor
        2. When using Adam, it seems word embedding is not trained on GPU.
            print(embd_var.name)
           Actually, CPU is not capable to execute 8-worker word embedding Adam updating, which
            cause the GPU usage-->0% and the train is very slow.
        3. We employ AdaGrad instead, if args.train_embd == True.
           Gradient clip is barely used in AdaGrad.
           Other optimizator like RMSProp, Momentum have not tested.

        ref: http://stackoverflow.com/questions/40621240/gpu-cpu-tensorflow-training
        ref: http://stackoverflow.com/questions/36498127/
                  how-to-effectively-apply-gradient-clipping-in-tensor-flow
        ref: http://stackoverflow.com/questions/35828037/
                  training-a-cnn-with-pre-trained-word-embeddings-is-very-slow-tensorflow

        To Solve the problem:
        ref: https://github.com/tensorflow/tensorflow/issues/6460
        '''

        # ------------- calc gradients --------------------------
        var_list = [embd_var] + other_var_list
        if grads is None:
            grads = tf.gradients(cost, var_list)

        # ------------- Optimization -----------------------
        # 0. global step used for asynchronous distributed training.
        # 1. Adam (default lr 0.0004 for 8 GPUs, 300 batchsize) if args.train_embd == False,
        #    apply gradient clip operations (default 10, 100)
        # 2. Adagrad (default 0.01~0.03? 1e-8? for 8 GPUs, 300 batchsize) if train embedding,
        #    no gradient clip.
        #    However, Adagrad is not suitable for large datasets.
        # 3. Momentum (default 0.001)
        # 4. RMSProp/Adadelta (default 0.001) is also OK......

        if not train_embd:
            # -------------------- clip all gradients--------------
            grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, max_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            word_grad, other_grads = [grads[0]], grads[1:]
            # -------------------- clip all gradients except word embedding gradient --------------
            other_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in other_grads]
            other_grads, _ = tf.clip_by_global_norm(other_grads, max_norm)
            grads = word_grad + other_grads  # keep the order
            # optimizer = tf.train.AdamOptimizer(args.lr)
            # optimizer = tf.train.RMSPropOptimizer(args.lr)
            optimizer = AsyncAdamOptimizer(learning_rate)

        if not hasattr(self, 'global_step'):
            self.init_global_step()

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool

    def training_op(self, cost, var_list,
            grad_clip=5.0,
            max_norm=200.0,
            learning_rate=0.01,
            grads=None,
            train_embd=True):

        # ------------- calc gradients --------------------------
        if grads is None:
            grads = tf.gradients(cost, var_list)

        for i, g in enumerate(grads):
            if g is None:
                print('WARNING: {} is not in the graph'.format(var_list[i].name))

        grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, max_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        if not hasattr(self, 'global_step'):
            self.init_global_step()

        return optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)  # a tf.bool



    def model_setup(self):
        args = self.args
        with tf.variable_scope(args.session_name):
            self.init_global_step()
            self._create_placeholders()
            self._inference_graph()

            self.pred = tf.argmax(logits, axis=1, name="predictions")
            self.prob = tf.nn.softmax(logits)
            labels = tf.reshape(self.label_plh, [-1])

            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)

            learning_rate = tf.train.exponential_decay(args.learning_rate, self.global_step,
                    args.decay_steps,
                    args.decay_rate,
                    staircase=True)
            self.train_op = self.training_op(self.loss_total,
                    tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=args.log_prefix),
                    grad_clip=args.grad_clip,
                    max_norm=args.max_norm,
                    train_embd=True,
                    learning_rate=args.learning_rate,)
            self._logger.info("Created RnnClassifier.")
            self._create_saver(args)
            self._logger.info('Created Saver')

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss_reg_l1', self.loss_reg_l1)
            tf.summary.scalar('loss_reg_diff', self.loss_reg_diff)
            tf.summary.scalar('loss_reg_sharp', self.loss_reg_sharp)
            tf.summary.scalar('loss_total', self.loss_total)
            self.merged = tf.summary.merge_all()


            batch_size = tf.shape(self.input_plh)[0]

