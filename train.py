from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import sys

import tensorflow as tf
import numpy as np
# if 'research' in sys.path:
import nets.resnet_v2 as resnet_v2
import nets.inception as inception
# from tensorflow.contrib.slim.nets import inception
import adamb_data_loader


# for intermittent eval
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
# from research.slim.datasets
from datasets import cifar10

slim = tf.contrib.slim

# Limiting to only one GPU. TODO make this a flag somehow...
os.environ["CUDA_VISIBLE_DEVICES"]="0"


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

cifar_classes = np.asarray(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])



def searchsortedtf(bin_matrix, idx_seed):
    return np.searchsorted(bin_matrix, idx_seed)


def _get_optimizer(opt):
    if opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    elif opt == 'sgdm':
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    return optimizer


def run_adaptive_training():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            tf.logging.info('Setup')

            tf_global_step = tf.train.get_or_create_global_step()
            p_images = tf.placeholder(tf.float32, shape=(None, image_size[FLAGS.dataset_name], image_size[FLAGS.dataset_name], channels[FLAGS.dataset_name]))
            p_labels = tf.placeholder(tf.int64, shape=(None, num_classes[FLAGS.dataset_name]))
            p_emb_idx = tf.placeholder(tf.int32, shape=(FLAGS.batch_size/2, ))
            p_assign_idx = tf.placeholder(tf.int32)

            # Data for eval. Currently hardcoded for cifar
            eval_data_provider = dataset_data_provider.DatasetDataProvider(cifar10.get_split('test','.'),
                    common_queue_capacity=2*FLAGS.batch_size, common_queue_min=FLAGS.batch_size)
            e_image, e_label = eval_data_provider.get(['image', 'label'])
            e_image = tf.to_float(e_image) # TODO this is a hack
            eval_images, eval_labels = tf.train.batch([e_image,e_label], 
                                            batch_size=FLAGS.batch_size, 
                                            num_threads=1,
                                            capacity=5*FLAGS.batch_size,
                                            allow_smaller_final_batch=True)
            
            with tf.device('/cpu:0'):
                dataloader = adamb_data_loader.adamb_data_loader(FLAGS.dataset_name, decay=FLAGS.decay, loss_scaling=FLAGS.loss_scaling)



            if FLAGS.model == 'resnet':
                with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                    logits, end_points = resnet_v2.resnet_v2_50(p_images,
                                                                num_classes[FLAGS.dataset_name],
                                                                is_training=True,
                                                                global_pool=True)
                    embeddings = end_points['global_pool']  # size (BATCH,1,1,2048) # this doesn't work tf1.4. using batch
                    predictions = end_points['predictions']
                    predictions = tf.argmax(predictions, 1)

            if FLAGS.model == 'inception':
                with slim.arg_scope(inception.inception_v1_arg_scope()):
                    print(num_classes)
                    logits, end_points = inception.inception_v1(p_images,
                                                                num_classes[FLAGS.dataset_name],
                                                                is_training=True,
                                                                global_pool=True)
                    embeddings = end_points['global_pool']
                    predictions = tf.argmax(logits, 1)

            embeddings = tf.squeeze(embeddings, [1, 2])
            tf.logging.debug("embeddings size: ", embeddings.shape)

            sample_losses = tf.losses.softmax_cross_entropy(logits=logits,
                                                            onehot_labels=p_labels)
            # sample_losses = tf.losses.sparse_softmax_cross_entropy(logits=logits,
            #                                                 labels=p_labels,
            #                                                 loss_collection=None)
            # tf.losses.sparse_softmax_cross_entropy(
            #     labels=p_labels, logits=logits, weights=1.0)

            # Try total loss with sample loss commented out. also something is
            # happening here that is making softmax super slow...



            # total_loss = tf.losses.get_total_loss()

            optimizer = _get_optimizer(FLAGS.opt)

            # train_op = optimizer.minimize(tf.reduce_mean(sample_losses), global_step=tf_global_step)  # sample_losses)
            train_op = optimizer.minimize(sample_losses, global_step=tf_global_step)  # sample_losses)
            # train_op = slim.learning.create_train_op(total_loss, optimizer)
            tf.logging.info('Model + training setup')

            embedding_list = tf.get_variable('EmbeddingList', shape=[num_train_samples[FLAGS.dataset_name], 2048],
                                             initializer=tf.random_normal_initializer(mean=0.3, stddev=0.6),
                                             trainable=False)

            b = tf.gather(embedding_list, p_emb_idx, axis=0)
            c = tf.matmul(b, embedding_list, transpose_b=True) # this transpose could be backwards
            squared_euclid = tf.transpose(tf.transpose(tf.reduce_sum(tf.square(embedding_list), axis=1) - 2*c) + tf.reduce_sum(tf.square(b), axis=1)) # TODO check this, last term may be incorrect

            if FLAGS.pot_func == 'sq_recip':
                recip_squared_euclid = tf.reciprocal(squared_euclid + FLAGS.recip_scale)  # hyperparam fix infs
                potential = recip_squared_euclid
            else:
                neg_exp_euclid = tf.exp(-FLAGS.recip_scale*squared_euclid/1000)
                potential = neg_exp_euclid

            m, n = potential.get_shape().as_list()
            class_starts = dataloader.class_starts

            def get_mask(class_starts, labels, batch_size):
                labels_mask = np.ones(shape=(batch_size, 50000))  # fix these hard codes
                static_range = 5000  # fix these hard codes
                # class_starts = np.asarray(class_starts)  # Possibly relevant
                mins = class_starts[labels]
                mask_idxs = mins[..., None] + np.arange(static_range)
                labels_mask[np.expand_dims(np.arange(batch_size), 1), mask_idxs] = 0.0
                return labels_mask

            labels_mask = tf.py_func(get_mask, [class_starts, tf.argmax(p_labels, axis=1),
                                                tf.cast(FLAGS.batch_size / 2, tf.int32)], tf.double)  #tf.int32)  # TODO last term may be wrong

            diverse_dist = tf.multiply(potential, tf.cast(labels_mask, tf.float32))

            cumm_array = tf.cumsum(diverse_dist, axis=1)
            max_cumm_array = tf.reduce_max(cumm_array, axis=1)
            bin_min = tf.cumsum(max_cumm_array + 1, exclusive=True)
            cumm_array = tf.expand_dims(bin_min, 1) + cumm_array
            scaled_seed = bin_min + tf.multiply(tf.random_uniform([tf.cast(FLAGS.batch_size/2, tf.int32), ]), max_cumm_array)
            scaled_seed_idx = tf.py_func(searchsortedtf, [tf.reshape(cumm_array, [-1]), scaled_seed], tf.int64)
            pair_idx_tensor = tf.cast(scaled_seed_idx, tf.int32) - tf.range(m) * n

            # Embedding update
            emb_update_op = tf.scatter_nd_update(embedding_list, tf.expand_dims(p_assign_idx, 1), embeddings)


            # predictions = tf.squeeze(end_points['predictions'], [1, 2], name='SpatialSqueeze')
            # predictions = tf.argmax(predictions, 1)
            accuracy_op = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.argmax(p_labels, 1))))

            tf.summary.scalar('Train_Accuracy', accuracy_op)
            tf.summary.scalar('Total_Loss', sample_losses) #total_loss)
            tf.summary.image('input', p_images)
            slim.summaries.add_histogram_summaries(slim.get_model_variables())
            # slim.summaries.add_histogram_summaries()
            summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)
            merged_summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()

            tf.logging.info('Savers')
            sess.run(tf.global_variables_initializer())

            # Training loop
            for step in range(FLAGS.max_steps):
                # TODO Still need to modify this to do more than just half-batches
                # TODO should be split up into functions that return images/labels from idx and a separate function that finds those idxs, not one in the same
                images, _, labels, sample_idxs = dataloader.load_batch(batch_size=FLAGS.batch_size, method=FLAGS.method)
                    # print('images', images.shape, 'labels', labels.shape, 'sample_idxs', sample_idxs.shape)

                if FLAGS.method == 'pairwise':
                    pair_idxs = sess.run(pair_idx_tensor, feed_dict={p_emb_idx: sample_idxs, p_labels: labels})
                    if FLAGS.debug:
                        np_diverse_dist = sess.run(diverse_dist, feed_dict={p_emb_idx: sample_idxs, p_labels: labels})
                        print('np_diverse_dist: ', np_diverse_dist)
                    image_pairs, label_pairs = dataloader.get_data_from_idx(pair_idxs)
                    images = np.concatenate((images, image_pairs), axis=0)
                    labels = np.concatenate((labels, label_pairs), axis=0)
                    sample_idxs = np.append(sample_idxs, pair_idxs)

                    _, losses, acc, batch_embeddings, _, summary = sess.run([train_op, sample_losses, accuracy_op, embeddings, emb_update_op, merged_summary_op],
                                                                       feed_dict={p_images: images, p_labels: labels, p_assign_idx: sample_idxs})
                    # _, losses, batch_embeddings, summary = sess.run([train_op, sample_losses, embeddings, merged_summary_op],
                    #                                                    feed_dict={p_images: images, p_labels: labels, p_assign_idx: sample_idxs})

                else:
                    _, losses, acc, summary = sess.run([train_op, sample_losses, accuracy_op, merged_summary_op],
                                                    feed_dict={p_images: images, p_labels: labels})

                tf.logging.debug('loss: ' + str(losses.mean()))

                if FLAGS.method == 'singleton':
                    dataloader.update(FLAGS.method, sample_idxs, metrics={'losses': losses})
                if FLAGS.method == 'pairwise':
                    dataloader.update(FLAGS.method, sample_idxs, metrics={'losses': losses, 'batch_embeddings': batch_embeddings})
                    
                if ((step + 1) % FLAGS.save_summary_iters == 0 or (step + 1) == FLAGS.max_steps):
                    tf.logging.info('Iteration %d complete', step)
                    summary_writer.add_summary(summary, step)
                    label_names=np.argmax(labels, 1)
                    # print(type(label_names))
                    # print(cifar_classes[label_names])
                    print(acc)
                    summary_writer.flush()
                    tf.logging.info('loss: ' + str(losses.mean()))
                    tf.logging.debug('Summary Saved')

                if ((step + 1) % FLAGS.save_model_iters == 0 or (step + 1) == FLAGS.max_steps):
                    checkpoint_file = join(FLAGS.train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=tf_global_step)
                    tf.logging.debug('Model Saved')

            #add an eval here at the end

def main(_):
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    run_adaptive_training()


if __name__ == '__main__':
    tf.app.run()
