from ray.rllib.models.model import Model
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import tensorflow.contrib.rnn as rnn


def add_time_dimension(padded_inputs, seq_lens):
    """Adds a time dimension to padded inputs.
    Arguments:
        padded_inputs (Tensor): a padded batch of sequences. That is,
            for seq_lens=[1, 2, 2], then inputs=[A, *, B, B, C, C], where
            A, B, C are sequence elements and * denotes padding.
        seq_lens (Tensor): the sequence lengths within the input batch,
            suitable for passing to tf.nn.dynamic_rnn().
    Returns:
        Reshaped tensor of shape [NUM_SEQUENCES, MAX_SEQ_LEN, ...].
    """

    # Sequence lengths have to be specified for LSTM batch inputs. The
    # input batch must be padded to the max seq length given here. That is,
    # batch_size == len(seq_lens) * max(seq_lens)
    padded_batch_size = tf.shape(padded_inputs)[0]
    max_seq_len = padded_batch_size // tf.shape(seq_lens)[0]

    # Dynamically reshape the padded batch to introduce a time dimension.
    new_batch_size = padded_batch_size // max_seq_len
    new_shape = ([new_batch_size, max_seq_len] +
                padded_inputs.get_shape().as_list()[1:])
    return tf.reshape(padded_inputs, new_shape)

class Pommberman(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]
        hiddens = [128]
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        vision_in = inputs['boards']
        metrics_in = inputs['states']

        with tf.name_scope("pommber_vision"):
            vision_in = tf.transpose(vision_in, [0, 2, 3, 1])
            vision_in = slim.conv2d(vision_in, 32, [3, 3], 1, scope="conv_1")
            vision_in = slim.conv2d(vision_in, 16, [3, 3], 1, scope="conv_2")
            vision_in = slim.conv2d(vision_in, 6, [3, 3], 1, scope="conv_3")
            vision_in = slim.flatten(vision_in)

        with tf.name_scope("pommber_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                12,
                weights_initializer=xavier_initializer(),
                activation_fn=activation,
                scope="metrics_out")

        with tf.name_scope("pommber_out"):
            last_layer = tf.concat([vision_in, metrics_in], axis=1)
            last_layer = slim.fully_connected(
                last_layer,
                128,
                weights_initializer=xavier_initializer(),
                activation_fn=activation,
                scope="middel")

            cell_size = 64
            last_layer = add_time_dimension(last_layer, self.seq_lens)
            lstm = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
            self.state_init = [
                np.zeros(lstm.state_size.c, np.float32),
                np.zeros(lstm.state_size.h, np.float32)
            ]
            if self.state_in:
                c_in, h_in = self.state_in
            else:
                c_in = tf.placeholder(
                    tf.float32, [None, lstm.state_size.c], name="c")
                h_in = tf.placeholder(
                    tf.float32, [None, lstm.state_size.h], name="h")
                self.state_in = [c_in, h_in]

            # Setup LSTM outputs
    
            state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_out, lstm_state = tf.nn.dynamic_rnn(
                lstm,
                last_layer,
                initial_state=state_in,
                sequence_length=self.seq_lens,
                time_major=False,
                dtype=tf.float32)

            self.state_out = list(lstm_state)

            last_layer = tf.reshape(lstm_out, [-1, cell_size])
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")

        return output, last_layer


ModelCatalog.register_custom_model("pommberman_lstm", Pommberman)
