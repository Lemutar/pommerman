from ray.rllib.models.model import Model
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer

class Pommberman(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]
        hiddens = [128, 128]
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        vision_in = inputs['boards']
        metrics_in = inputs['states']

        with tf.name_scope("pommber_vision"):
            vision_in = tf.transpose(vision_in, [0, 2, 3, 1])
            vision_in = slim.conv2d(vision_in, 32, [3, 3],1, scope="conv_1")
            vision_in = slim.conv2d(vision_in, 6, [5, 5],1, scope="conv_2")
            vision_in = slim.flatten(vision_in)

        with tf.name_scope("pommber_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                24,
                weights_initializer=xavier_initializer(),
                activation_fn=activation,
                scope="metrics_out")

        with tf.name_scope("pommber_out"):
            i = 1
            last_layer = tf.concat([vision_in, metrics_in], axis=1)
            for size in hiddens:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=xavier_initializer(),
                    activation_fn=activation,
                    scope="fc{}".format(i))
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")

        return output, last_layer


ModelCatalog.register_custom_model("pommberman", Pommberman)
