import tensorflow as tf
import logging


class CNNLSTM():
    def __init__(self, args):
        self._logger = logging.getLogger(__name__)

    def _create_placeholders(self, args):
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



    def init_global_step(self):
        # Global steps for asynchronous distributed training.
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable('global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)


    def model_setup(self, args):
        with tf.variable_scope(args.session_name):
            self.init_global_step()
            self._create_placeholders(args)
            self._create_cnn_net(args)
            


            batch_size = tf.shape(self.input_plh)[0]
              

