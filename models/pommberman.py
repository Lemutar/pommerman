from ray.rllib.models.model import Model
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.misc import normc_initializer

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer


class Pommberman(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]
        hiddens = [256]
        activation = tf.nn.relu

        vision_in = inputs['boards']
        metrics_in = inputs['states']

        with tf.name_scope("pommber_vision"):
            vision_in = tf.transpose(vision_in, [0, 2, 3, 1])
            vision_in = slim.conv2d(vision_in, 32, 3, 1, padding="valid", scope="conv_1")
            vision_in = slim.conv2d(vision_in, 64, 3, 1, padding="valid", scope="conv_2")
            vision_in = slim.conv2d(vision_in, 128, 3, 1, padding="valid", scope="conv_3")
            vision_in = slim.conv2d(vision_in, 256, 3, 1, padding="valid", scope="conv_4")
            vision_in = tf.layers.flatten(vision_in)


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
