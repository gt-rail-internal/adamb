from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import adamb_data_loader

tf.app.flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
tf.app.flags.DEFINE_string('f', '', 'kernel')  # Hack because for some reason tf.app.flags produces errors without it

tf.app.flags.DEFINE_string('method', 'singleton', 'technique for constructing the minibatch.')
tf.app.flags.DEFINE_string('dataset_name', 'cifar10', 'name of the dataset.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of images in each eval branch.')
tf.app.flags.DEFINE_integer('eval_batch_size', 256, 'The number of images in each branch.')
tf.app.flags.DEFINE_integer('max_steps', 2500, 'The maximum number of gradient steps.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'The step size of gradient descent.')
tf.app.flags.DEFINE_string('train_log_dir', '/tmp/data_adamb', 'Directory where to write stuff.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 15, 'The summary save period.')
tf.app.flags.DEFINE_integer('save_summary_iters', 100, 'The summary save frequency.')
tf.app.flags.DEFINE_integer('save_model_iters', 200, 'The model save frequency.')
tf.app.flags.DEFINE_string('hparams', '', 'String of key-valued pair hyperparameters.')
tf.app.flags.DEFINE_float('decay', 0.99, 'The decay for expoential moving average.')
tf.app.flags.DEFINE_float('loss_scaling', 100, 'The loss_scaling for exponential regularization for singletons.')
tf.app.flags.DEFINE_string('model', 'resnet', 'model name.')
tf.app.flags.DEFINE_float('recip_scale', 1.0, 'scaleing value for the potential function.')
tf.app.flags.DEFINE_string('pot_func', 'sq_recip', 'type of potential function.')

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


def get_hparams(flag_string=None):

    # hparams = tf.HParams(opt='sgdm',
    hparams = tf.contrib.training.HParams(opt='sgdm',
                         method=FLAGS.method,
                         decay=FLAGS.decay, 
                         learning_rate=FLAGS.learning_rate,
                         model=FLAGS.model,
                         loss_scaling=FLAGS.loss_scaling,
                         batch_size=FLAGS.batch_size,
                         max_steps=FLAGS.max_steps)
    if flag_string:
        hparams.parse(flag_string)
    return hparams


def searchsortedtf(bin_matrix, idx_seed):
    return np.searchsorted(bin_matrix, idx_seed)


def run_adaptive_training(hparams):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if FLAGS.debug:
                print('Started')

            p_images = tf.placeholder(tf.float32, shape=(None, image_size[FLAGS.dataset_name], image_size[FLAGS.dataset_name], channels[FLAGS.dataset_name]))
            p_labels = tf.placeholder(tf.int64, shape=(None, num_classes[FLAGS.dataset_name]))
            p_emb_idx = tf.placeholder(tf.int32, shape=(hparams.batch_size/2, ))
            p_assign_idx = tf.placeholder(tf.int32)

            if FLAGS.debug:
                print("Placeholders")
                
            with tf.device('/cpu:0'):
                dataloader = adamb_data_loader.adamb_data_loader(FLAGS.dataset_name, decay=hparams.decay, loss_scaling=hparams.loss_scaling)

            if hparams.model == 'resnet':
                with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
                    logits, end_points = nets.resnet_v2.resnet_v2_50(p_images,
                                                                    num_classes[FLAGS.dataset_name],
                                                                    is_training=True,
                                                                    global_pool=True)
                # embeddings = end_points['global_pool']  # size (BATCH,1,1,2048) # this doesn't work tf1.4. using batch
                embeddings = end_points['resnet_v2_50/block4']  # TODO might need a global_pool layer added afterwards.


            print("embeddings size: ", embeddings.shape)

            # Python 2
            # for key, value in end_points.iteritems():
            #     print(key)

            # Python 3
            for key, value in end_points.items():
                print(key)

            embedding_list = tf.get_variable('EmbeddingList', shape=[num_train_samples[FLAGS.dataset_name], 2048],
                                             initializer=tf.random_normal_initializer(mean=0.3, stddev=0.6),
                                             trainable=False)

            predictions = tf.squeeze(end_points['predictions'], [1, 2], name='SpatialSqueeze')
            predictions = tf.argmax(predictions, 1)
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.argmax(p_labels, 1))))
            sample_losses = tf.losses.softmax_cross_entropy(logits=logits[:,0,0,:], onehot_labels=p_labels, 
                                            reduction=tf.losses.Reduction.NONE, loss_collection=None)
            total_loss = tf.losses.get_total_loss()
            
            tf_global_step = tf.train.get_or_create_global_step()

            if hparams.opt == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
            elif hparams.opt == 'sgdm':
                optimizer = tf.train.MomentumOptimizer(learning_rate=hparams.learning_rate, momentum=0.9)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=hparams.learning_rate)

            train_op = slim.learning.create_train_op(total_loss, optimizer)
            if FLAGS.debug:
                print('Model + training setup')

            embeddings = tf.squeeze(embeddings, [1, 2])

            b = tf.gather(embedding_list, p_emb_idx, axis=0)
            c = tf.matmul(b, embedding_list, transpose_b=True) # this transpose could be backwards
            squared_euclid = tf.transpose(tf.transpose(tf.reduce_sum(tf.square(embedding_list), axis=1) - 2*c) + tf.reduce_sum(tf.square(b), axis=1)) # TODO check this, last term may be incorrect
            recip_squared_euclid = tf.reciprocal(squared_euclid+FLAGS.recip_scale)

            if FLAGS.pot_func == 'sq_recip':
                recip_squared_euclid = tf.reciprocal(squared_euclid + FLAGS.recip_scale) # hyperparam fix infs
                potential = recip_squared_euclid
            else:
                neg_exp_euclid = tf.exp(-FLAGS.recip_scale*squared_euclid/1000)
                potential = neg_exp_euclid

            m, n = potential.get_shape().as_list()  # TODO should this be .get_shape.as_list() ?
            
            # TODO this diversity measure simply ensures that its not the same class as the seed. Too naive?

            class_starts = dataloader.class_starts
            non_one_hot = tf.argmax(p_labels, axis=1)

            def get_mask(class_starts, labels, batch_size):
                labels_mask = np.ones(shape=(batch_size, 50000))  # fix these hard codes
                static_range = 5000  # fix these hard codes
                # class_starts = np.asarray(class_starts)  # Possibly relevant
                mins = class_starts[labels]
                mask_idxs = mins[..., None] + np.arange(static_range)
                labels_mask[np.expand_dims(np.arange(batch_size), 1), mask_idxs] = 0.0
                return labels_mask

            labels_mask = tf.py_func(get_mask, [class_starts, tf.argmax(p_labels, axis=1),
                                                tf.cast(hparams.batch_size / 2, tf.int32)], tf.double)  #tf.int32)  # TODO last term may be wrong

            diverse_dist = tf.multiply(potential, tf.cast(labels_mask, tf.float32))

            cumm_array = tf.cumsum(diverse_dist, axis=1)
            max_cumm_array = tf.reduce_max(cumm_array, axis=1)
            bin_min = tf.cumsum(max_cumm_array + 1, exclusive=True)
            cumm_array = tf.expand_dims(bin_min, 1) + cumm_array
            scaled_seed = bin_min + tf.multiply(tf.random_uniform([tf.cast(hparams.batch_size/2, tf.int32), ]), max_cumm_array)
            scaled_seed_idx = tf.py_func(searchsortedtf, [tf.reshape(cumm_array, [-1]), scaled_seed], tf.int64)
            pair_idx_tensor = tf.cast(scaled_seed_idx, tf.int32) - tf.range(m) * n

            # Embedding update
            assign_op = tf.scatter_nd_update(embedding_list, tf.expand_dims(p_assign_idx, 1), embeddings)

            tf.summary.scalar('Train_Accuracy', accuracy)
            tf.summary.scalar('Total_Loss', total_loss)
            slim.summaries.add_histogram_summaries(slim.get_model_variables())
            summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)
            merged_summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            
            if FLAGS.debug:
                print('Savers')
            sess.run(tf.global_variables_initializer())

            # Training loop
            
            cumm_mean = 0
            cumm_stf = 0
            
            for step in range(FLAGS.max_steps): 
                # TODO Still need to modify this to do more than just half-batches
                # TODO should be split up into functions that return images/labels from idx and a separate function that finds those idxs, not one in the same
                images, _, labels, sample_idxs = dataloader.load_batch(batch_size=hparams.batch_size, method=hparams.method)
                # height = image_size[FLAGS.dataset_name],
                # width = image_size[FLAGS.dataset_name],

                if FLAGS.debug:
                    print('images', images.shape, 'labels', labels.shape, 'sample_idxs', sample_idxs.shape)

                if FLAGS.debug:
                    print('Model + training setup')

                if hparams.method == 'pairwise':
                    pair_idxs = sess.run(pair_idx_tensor, feed_dict={p_emb_idx: sample_idxs, p_labels: labels})
                    if FLAGS.debug:
                        np_diverse_dist = sess.run(diverse_dist, feed_dict={p_emb_idx: sample_idxs, p_labels: labels})
                        print('np_diverse_dist: ', np_diverse_dist)
                    image_pairs, label_pairs = dataloader.get_data_from_idx(pair_idxs)
                    images = np.concatenate((images, image_pairs), axis=0)
                    labels = np.concatenate((labels, label_pairs), axis=0)
                    sample_idxs = np.append(sample_idxs, pair_idxs)

                    # print("image_pairs ", image_pairs.shape)
                    # print("images ", images.shape)
                    # print("labels ", labels.shape)
                    # print("sample_idxs ", sample_idxs.shape)
                    #
                    # flat_sample_idxs = sample_idxs.reshape[-1]

                    # _, losses, batch_embeddings, _, summary = sess.run([train_op, sample_losses, embeddings, assign_op, merged_summary_op],
                    #                                                     feed_dict={p_images: images, p_labels: labels, p_assign_idx: sample_idxs})
                    _, losses, batch_embeddings, summary = sess.run([train_op, sample_losses, embeddings, merged_summary_op],
                                                                        feed_dict={p_images: images, p_labels: labels, p_assign_idx: sample_idxs})

                    if FLAGS.debug:
                        print('losses: ', np.mean(losses))
                        print('batch_embeddings: ', batch_embeddings.shape)

                else:
                    _, losses, summary = sess.run([train_op, sample_losses, merged_summary_op],
                                                    feed_dict={p_images: images, p_labels: labels})

                if FLAGS.debug:
                    print('Ran!')

                if hparams.method == 'singleton':
                    if FLAGS.debug:
                        print('losses', losses.shape, 'sample_idxs', sample_idxs.shape)
                    dataloader.update(hparams.method, sample_idxs, metrics={'losses': losses})
                if hparams.method == 'pairwise':
                    dataloader.update(hparams.method, sample_idxs, metrics={'losses': losses, 'batch_embeddings': batch_embeddings})
                    
                if ((step + 1) % FLAGS.save_summary_iters == 0 or (step + 1) == FLAGS.max_steps):
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                    # checkpoint_file = os.path.join(FLAGS.train_log_dir, 'model.ckpt')
                    # saver.save(sess, checkpoint_file, global_step=tf_global_step)

                if ((step + 1) % FLAGS.save_model_iters == 0 or (step + 1) == FLAGS.max_steps):
                    checkpoint_file = os.path.join(FLAGS.train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=tf_global_step)


def main(argv):
    if len(argv) > 1:
        raise tf.app.UsageError('Too many command-line arguments.')
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = get_hparams(FLAGS.hparams)
    run_adaptive_training(hparams)


if __name__ == '__main__':
    tf.app.run(main)