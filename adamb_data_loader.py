import numpy as np
import tensorflow as tf
# import cifar10

# from PIL import Image, ImageOps, ImageEnhance
# import Image, ImageOps, ImageEnhance

num_train_samples = {'cifar10': 50000,
                            'mnist': 50000,
                            'imagenet': 1000000}

num_classes = {'cifar10': 10,
                'mnist': 10,
                'imagenet': 1001}

class adamb_data_loader():

    def __init__(self, dataset_name, decay=0.9, bin_init=1.0, loss_scaling=100):
        self.dataset_name = dataset_name
        self.decay = decay
        self.num_train_samples = num_train_samples[self.dataset_name]
        self.loss_scaling = loss_scaling

        # Keras dataset import is tuple of two elements, [0] data (samples, imheight, imwidth, channels)
        # [1] labels (samples,1)
        if self.dataset_name == 'cifar10':
            train_set, test_set = tf.keras.datasets.cifar10.load_data()
        elif self.dataset_name == 'mnist':
            train_set, test_set = tf.keras.datasets.mnist.load_data()

        # sorting by label to enforce diversity
        train_labels_sort = train_set[1].reshape(-1).argsort(kind='heapsort')  # doesn't need to be heapsort
        self.train_labels = train_set[1][train_labels_sort].reshape(-1)
        self.train_data = train_set[0][train_labels_sort]

        self.class_starts = np.searchsorted(self.train_labels, np.arange(10))

        self.singletons = np.linspace(start=bin_init, stop=bin_init*self.num_train_samples, num=self.num_train_samples)
        self.embeddings = np.random.normal(0.3, 0.6, size=self.num_train_samples)

    def load_batch(self, batch_size=16, one_hot=True, method='singleton', split='train'):
        if split=='test':
            pass
        else:
            if method == 'singleton':
                bin_samples = self.singletons[-1]*np.random.random_sample(size=batch_size)
                sample_idxs = np.searchsorted(self.singletons, bin_samples)

            elif method == 'pairwise':
                # # Seed from singletons
                # bin_samples = self.singletons[-1]*np.random.random_sample(size=int(batch_size/2))
                # sample_idxs = np.searchsorted(self.singletons, bin_samples)
                # Seed from random
                sample_idxs = np.random.randint(self.num_train_samples, size=batch_size)

            else:
                # Select images randomly
                sample_idxs = np.random.randint(self.num_train_samples, size=batch_size)
            images, labels = self.get_data_from_idx(sample_idxs, one_hot)
            images_raw = np.expand_dims(images, 0)
            images_raw = np.squeeze(images_raw)

        

        return images, images_raw, labels, sample_idxs

    def _get_one_hot(self, labels):
        one_hot_matrix = np.zeros((labels.shape[0], num_classes[self.dataset_name]))
        one_hot_matrix[np.arange(labels.shape[0]), labels] = 1
        return one_hot_matrix

    def update(self, method, sample_idxs, metrics):
        if method == 'singleton':
            self.update_singletons(sample_idxs, losses=metrics['losses'])
        elif method == 'pairwise':
            pass  # TODO fix this when embeddings are separated
        else:
            print('No update associated with method: ', method)

    def update_singletons(self, sample_idxs, losses):
        for sample, loss in zip(sample_idxs, losses):
            self.singletons[sample:] += self.decay*(np.exp(loss/self.loss_scaling)-(self.singletons[sample]-self.singletons[sample-1]))  # TODO this is wrong. fix from notes. last term and exponent should apply to entire update

    def get_data_from_idx(self, sample_idxs, one_hot=True):
        images = self.train_data[sample_idxs]
        labels = np.squeeze(self.train_labels[sample_idxs])
        if one_hot:
            labels = self._get_one_hot(labels)
        return images, labels
