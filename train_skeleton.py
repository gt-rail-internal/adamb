from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join

import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim.nets as nets
import adamb_data_loader_skeleton as data_loader
from tensorflow.examples.tutorials.mnist import mnist
import re
import tensorflow.contrib.slim as slim

# for testing only
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets



tf.app.flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
tf.app.flags.DEFINE_string('f', '', 'kernel')  # Hack because for some reason tf.app.flags produces errors without it

tf.app.flags.DEFINE_string('dataset_name', 'cifar10', 'name of the dataset.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of images in each eval branch.')
tf.app.flags.DEFINE_float('decay', 0.95, 'The decay for expoential moving average.')
tf.app.flags.DEFINE_float('loss_scaling', 50, 'The loss_scaling for exponential regularization for singletons.')
tf.app.flags.DEFINE_string('method', 'pairwise', 'technique for constructing the minibatch.')
# tf.app.flags.DEFINE_integer('eval_batch_size', 256, 'The number of images in each branch.')
tf.app.flags.DEFINE_string('model', 'resnet', 'model name.')
tf.app.flags.DEFINE_float('recip_scale', 1.0, 'scaling value for the potential function. must be  > 0.')
tf.app.flags.DEFINE_string('pot_func', 'sq_recip', 'type of potential function.')
tf.app.flags.DEFINE_string('opt', 'sgdm', 'the optimization algorithm used')
tf.app.flags.DEFINE_integer('max_steps', 2000, 'The maximum number of gradient steps.')
tf.app.flags.DEFINE_float('learning_rate', 0.03, 'The step size of gradient descent.')
tf.app.flags.DEFINE_string('train_log_dir', '/tmp/data_adamb', 'Directory where to write stuff.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 25, 'The summary save period.')
tf.app.flags.DEFINE_integer('save_summary_iters', 150, 'The summary save frequency.')
tf.app.flags.DEFINE_integer('save_model_iters', 300, 'The model save frequency.')

FLAGS = tf.app.flags.FLAGS


image_size = {'cifar10': 32,
              'mnist': 28,
              'imagenet': 224}
              
channels = {'cifar10': 3,
            'mnist': 1,
            'imagenet': 3}

num_train_samples = {'cifar10': 50000,
                     'mnist': 50000,
                     'imagenet': 1000000}

num_classes = {'cifar10': 10,
               'mnist': 10,
               'imagenet': 1000}

TOWER_NAME = 'tower'

NUM_CLASSES = 10


def searchsortedtf(bin_matrix, idx_seed):
    return np.searchsorted(bin_matrix, idx_seed)


def _get_optimizer():
    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    elif FLAGS.opt == 'sgdm':
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    return optimizer

def _activation_summary(x):
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def inference(images):
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    dtype = tf.float32
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2), dtype=dtype)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2), dtype=dtype)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # print(images.get_shape()[0])
        print(pool2.get_shape())
        reshape = tf.reshape(pool2, [None, -1]) #[images.get_shape().as_list()[0], -1])
        print(images.get_shape())
        print(reshape.get_shape())
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=0.04), dtype=dtype)
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=0.04), dtype=dtype)
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[192, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=1 / 192.0), dtype=dtype)
        biases = tf.get_variable('biases', [NUM_CLASSES], initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def run_adaptive_training():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            tf.logging.info('Setup')

            tf_global_step = tf.train.get_or_create_global_step()
            if FLAGS.dataset_name == 'mnist':
                mnist_data = read_data_sets("MNIST_data/", one_hot=True)
            # Keras dataset import is tuple of two elements, [0] data (samples, imheight, imwidth, channels)
            # [1] labels (samples,1)
            if FLAGS.dataset_name == 'cifar10':
                train_set, _ = tf.keras.datasets.cifar10.load_data()
                data_shape = list(train_set[0][0].shape)
                label_shape = list(train_set[1][0].shape)

            IMAGE_PIXELS = np.prod(np.array(data_shape))  # image_size[FLAGS.dataset_name]
            # NUM_CLASSES = 10  # num_classes[FLAGS.dataset_name]

            p_images = tf.placeholder(tf.float32, shape=([None]+data_shape))
            p_labels = tf.placeholder(tf.int64, shape=(None, num_classes[FLAGS.dataset_name]))

            # with tf.device('/cpu:0'):
            # dataloader = data_loader.adamb_data_loader(FLAGS.dataset_name,
            #                                                  decay=FLAGS.decay,
            #                                                  loss_scaling=FLAGS.loss_scaling)

            # data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

            #############
            #TESTING ONLY
            #############
            # input x - for 28 x 28 pixels = 784
            x = tf.placeholder(tf.float32, [None, 784])
            # now declare the output data placeholder - 10 digits
            y = tf.placeholder(tf.float32, [None, 10])

            #############
            #############

            # if FLAGS.model == 'resnet':
            #     with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
            #         logits, end_points = nets.resnet_v2.resnet_v2_50(p_images,
            #                                                         num_classes[FLAGS.dataset_name],
            #                                                         is_training=True)
            #         # logits, end_points = nets.resnet_v2.resnet_v2_50(x,
            #         #                                                 num_classes[FLAGS.dataset_name],
            #         #                                                 is_training=True)
            # if FLAGS.model == 'inception':
            #     scope = 'InceptionV1'
            #     with tf.variable_scope(scope, 'InceptionV1', [p_images]) as scope:
            #         with slim.arg_scope([tf.layers.dropout], is_training=True):
            #             with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
            #                 net, end_points = nets.inception.inception_v1_base(p_images, scope=scope)
            #     # with slim.arg_scope(nets.inception.inception_v1_arg_scope(), is_training=True):
            #     #     net, end_points = nets.inception.inception_v1_base(p_images,
            #     #                                                        num_classes[FLAGS.dataset_name]) #,
            #                                                            #is_training=True)

            ############
            ## Dummy net
            ############
            #
            #
            # hidden1_units = 128
            # hidden2_units = 32
            #
            #
            # with tf.name_scope('hidden1'):
            #     weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
            #                           stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
            #     biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
            #     hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
            # # Hidden 2
            # with tf.name_scope('hidden2'):
            #     weights = tf.Variable(
            #         tf.truncated_normal([hidden1_units, hidden2_units],
            #                             stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
            #     biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
            #     hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
            # # Linear
            # with tf.name_scope('softmax_linear'):
            #     weights = tf.Variable(
            #         tf.truncated_normal([hidden2_units, NUM_CLASSES],
            #                             stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
            #     biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
            #     logits = tf.matmul(hidden2, weights) + biases
            #
            #
            #############
            #############
            logits = inference(p_images)

            sample_losses = tf.losses.softmax_cross_entropy(logits=logits, #[:,0,0,:],
                                                            onehot_labels=p_labels, #y, #p_labels,
                                                            reduction=tf.losses.Reduction.NONE,
                                                            loss_collection=None)

            # total_loss = tf.losses.get_total_loss()
            optimizer = _get_optimizer()
            # train_op = slim.learning.create_train_op(total_loss, optimizer)
            train_op = optimizer.minimize(tf.reduce_mean(sample_losses), global_step=tf_global_step)
            tf.logging.info('Model + training setup')

            # predictions = tf.squeeze(end_points['predictions'], [1, 2], name='SpatialSqueeze')
            # predictions = tf.argmax(predictions, 1)
            predictions = tf.argmax(logits, 1)
            # accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.argmax(p_labels, 1))))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.argmax(y, 1))))
            tf.summary.scalar('Train_Accuracy', accuracy)
            # tf.summary.scalar('Total_Loss', total_loss)
            # tf.summary.image('input', p_images)
            slim.summaries.add_histogram_summaries(slim.get_model_variables())
            # slim.summaries.add_histogram_summaries(gradients)
            summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)
            merged_summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            
            if FLAGS.debug:
                print('Savers')
            sess.run(tf.global_variables_initializer())

            # Training loop
            for step in range(FLAGS.max_steps):
                # TODO Still need to modify this to do more than just half-batches
                # TODO should be split up into functions that return images/labels from idx and a separate function that finds those idxs, not one in the same
                # images, _, labels, sample_idxs = dataloader.load_batch(batch_size=FLAGS.batch_size, method=FLAGS.method)
                    # print('images', images.shape, 'labels', labels.shape, 'sample_idxs', sample_idxs.shape)

                images_feed, labels_feed = mnist_data.train.next_batch(FLAGS.batch_size, False)

                # print(type(images_feed.shape[0:]), labels_feed.shape)

                # if FLAGS.method == 'pairwise':
                #     pair_idxs = sess.run(pair_idx_tensor, feed_dict={p_emb_idx: sample_idxs, p_labels: labels})
                #     if FLAGS.debug:
                #         np_diverse_dist = sess.run(diverse_dist, feed_dict={p_emb_idx: sample_idxs, p_labels: labels})
                #         print('np_diverse_dist: ', np_diverse_dist)
                #     image_pairs, label_pairs = dataloader.get_data_from_idx(pair_idxs)
                #     images = np.concatenate((images, image_pairs), axis=0)
                #     labels = np.concatenate((labels, label_pairs), axis=0)
                #     sample_idxs = np.append(sample_idxs, pair_idxs)
                #
                #     # _, losses, batch_embeddings, _, summary = sess.run([train_op, sample_losses, embeddings, emb_update_op, merged_summary_op],
                #     #                                                     feed_dict={p_images: images, p_labels: labels, p_assign_idx: sample_idxs})
                #     _, losses, batch_embeddings, summary = sess.run([train_op, sample_losses, embeddings, merged_summary_op],
                #                                                         feed_dict={p_images: images, p_labels: labels, p_assign_idx: sample_idxs})
                #
                # else:
                # _, losses, summary = sess.run([train_op, sample_losses, merged_summary_op],
                #                                     feed_dict={p_images: images, p_labels: labels})
                _, summary = sess.run([train_op, merged_summary_op],
                                                    feed_dict={x: images_feed, y: labels_feed})

                # tf.logging.debug('loss: ' + str(losses.mean()))

                if ((step + 1) % FLAGS.save_summary_iters == 0 or (step + 1) == FLAGS.max_steps):
                    tf.logging.info('Iteration %d complete', step)
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                    # tf.logging.info('loss: ' + str(losses.mean()))
                    tf.logging.debug('Summary Saved')

                if ((step + 1) % FLAGS.save_model_iters == 0 or (step + 1) == FLAGS.max_steps):
                    checkpoint_file = join(FLAGS.train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=tf_global_step)
                    tf.logging.debug('Model Saved')



def main(_):
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    run_adaptive_training()


if __name__ == '__main__':
    tf.app.run()