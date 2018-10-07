
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


def compute_embedding_inception_v1(inputs,
                                   embedding_dim=64,
                                   is_training=True,
                                   dropout_keep_prob=0.8,
                                   scope='InceptionV1',
                                   l2_normalize=True):
    """Compute embedding with inception v1."""
    with tf.variable_scope(scope, 'InceptionV1', [inputs,
                                                  embedding_dim]) as scope:
        with slim.arg_scope(
            [layers.batch_norm, layers.dropout], is_training=is_training):

            with slim.arg_scope(inception.inception_v1_arg_scope()):
                net, end_points = inception.inception_v1_base(
                    inputs, scope=scope)
                net = layers.avg_pool2d(
                    net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
                net = layers.dropout(net, dropout_keep_prob, scope='Dropout_0b')

            base_variables = slim.get_variables_to_restore(
                exclude=['global_step'])

            # Embedding bottleneck.
            net = layers.conv2d(
                net,
                embedding_dim, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='Bottleneck')
            embedding = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            if l2_normalize:
                embedding = tf.nn.l2_normalize(embedding, dim=1)
            end_points['embeddings'] = embedding

            bottleneck_variables = tf.contrib.framework.get_variables(
                scope='InceptionV1/Bottleneck')

    return embedding, end_points, base_variables, bottleneck_variables


compute_embedding_inception_v1.default_image_size = 224


def get_embedding_fn(model_name):
    """Factory function of embedding model."""
    if model_name == 'inception_v1':
        return compute_embedding_inception_v1
    raise ValueError("Unkown embedding model: %s" % model_name)
