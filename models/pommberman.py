from ray.rllib.models.model import Model
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer

class Pommberman(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]
        convs = [
            [121, [1, 1], 4],
            [121, [1, 1], 3],
            [121, [1, 1], 2],
            [121, [1, 1], 1],
        ]
        hiddens = [256, 256]
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        vision_in = inputs['boards']
        metrics_in = inputs['states']

        with tf.name_scope("pommber_vision"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                vision_in = slim.conv2d(
                    vision_in,
                    out_size,
                    kernel,
                    stride,
                    scope="conv{}".format(i))
            out_size, kernel, stride = convs[-1]
            vision_in = slim.conv2d(
                vision_in,
                out_size,
                kernel,
                stride,
                padding="VALID",
                scope="conv_out")
            vision_in = tf.squeeze(vision_in, [1, 2])

        with tf.name_scope("pommber_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                64,
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


ModelCatalog.register_custom_model("Pommberman", Pommberman)