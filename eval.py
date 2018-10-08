from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
# import tensorflow.contrib.slim.nets as nets
import nets.resnet_v2 as resnet_v2
import nets.inception as inception
import math
from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from datasets import cifar10
import adamb_data_loader

slim = tf.contrib.slim

# Limits to only one GPU:
os.environ["CUDA_VISIBLE_DEVICES"]="0"


tf.app.flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

tf.app.flags.DEFINE_string('method', 'singleton', 'technique for constructing the minibatch.')
tf.app.flags.DEFINE_string('dataset_name', 'cifar10', 'name of the dataset.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of images in each eval branch.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'The step size of gradient descent.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/clean_adamb', 'Directory where checkpoints are.')
tf.app.flags.DEFINE_string('log_dir', '/tmp/clean_adamb', 'Directory where logs are .')
tf.app.flags.DEFINE_string('model', 'resnet', 'model name.')
tf.app.flags.DEFINE_string('master', '', 'The BNS address of the TensorFlow master, empty for nonborg.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 25, 'How often to check for/eval new checkpoint.')

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


def main(_):
    g=tf.Graph()
    with g.as_default():
        # dataset = datasets.get_dataset(FLAGS.dataset_name, 'test')
        data_provider = dataset_data_provider.DatasetDataProvider(cifar10.get_split('test','.'),
                common_queue_capacity=2*FLAGS.batch_size, common_queue_min=FLAGS.batch_size)
        # data_provider = dataset_data_provider.DatasetDataProvider(
        #    dataset, common_queue_capacity=2*FLAGS.batch_size, common_queue_min=FLAGS.batch_size)
        image, label = data_provider.get(['image', 'label'])
        image = tf.to_float(image) # TODO this is a hack
        images, labels = tf.train.batch([image,label], 
                                        batch_size=FLAGS.batch_size, 
                                        num_threads=1,
                                        capacity=5*FLAGS.batch_size,
                                        allow_smaller_final_batch=True)

        if FLAGS.model == 'resnet':
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, end_points = resnet_v2.resnet_v2_50(images,
                                                            num_classes[FLAGS.dataset_name],
                                                            is_training=True,
                                                            global_pool=True)
                predictions = end_points['predictions']
                predictions = tf.argmax(predictions, 1)

        if FLAGS.model == 'inception':
            with slim.arg_scope(inception.inception_v1_arg_scope()):
                print(num_classes)
                logits, end_points = inception.inception_v1(images,
                                                            num_classes[FLAGS.dataset_name],
                                                            is_training=True,
                                                            global_pool=True)
                predictions = tf.argmax(logits, 1)

        one_hot_labels = slim.one_hot_encoding(labels, num_classes[FLAGS.dataset_name])
        # one_hot_labels = tf.squeeze(one_hot_labels, axis=1)

        # Defining metrics:
        names_to_values, name_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': tf.metrics.accuracy(predictions=predictions, labels=labels),
            'Recall_5': tf.metrics.recall_at_k(predictions=end_points['predictions'],
                labels=tf.to_int64(one_hot_labels), k=5) 
            }) #TODO k is hardcoded
        
        for name, value in names_to_values.items():
            slim.summaries.add_scalar_summary(
                value, name, prefix='eval', print_summary=True)

        num_batches = math.ceil(num_train_samples[FLAGS.dataset_name] / float(FLAGS.batch_size)) 

        slim.evaluation.evaluation_loop(
                master=FLAGS.master,
                checkpoint_dir=FLAGS.checkpoint_dir,
                logdir=FLAGS.log_dir,
                num_evals=num_batches,
                eval_op=name_to_updates.values(),
                final_op=names_to_values.values(),
                eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()
