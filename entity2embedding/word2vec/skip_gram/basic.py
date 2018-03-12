# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

logger = logging.getLogger(__name__)


class BasicWord2vec(object):
    def __init__(self, vocabulary_size, index_to_word_map, embedding_size=300, num_sampled=20):
        """

        :param vocabulary_size:
        :param index_to_word_map:
        :param vocabulary_count:
        :param embedding_size: 300 features is what Google used in their published model trained on the Google news dataset
        :param num_sampled: The paper says that selecting 5-20 words works well for smaller datasets, and you can get away with only 2-5 words for large datasets.
        """
        self._vocabulary_size = vocabulary_size
        self._index_to_word_map = index_to_word_map
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled

        self._init_setup_global_step()
        self._init_setup_variable_for_export()

        # tensor
        self._word_embeddings = None
        self._normalized_embeddings = None
        self._nce_weights = None

        # op
        self.loss_op = None
        self.train_op = None

        #
        self.training_input = None
        self.labels = None

    def _init_setup_global_step(self):
        """setup global step tensor"""
        tf.contrib.framework.get_or_create_global_step()

    def _init_setup_variable_for_export(self):
        """setup some variable that will needed by export function"""
        word_list = list(map(lambda x: self._index_to_word_map[x],
                             range(self._vocabulary_size)))
        tf.Variable(
            np.array(word_list).reshape((-1, 1)),
            trainable=False,
            name="word_array"
        )

        tf.Variable(
            self.embedding_size,
            trainable=False,
            name="embedding_size"
        )

        tf.Variable(
            self._vocabulary_size,
            trainable=False,
            name="vocabulary_size"
        )

    def inference(self, x):
        """
        forward calculation from x to y
        :param x: tensor represent for input X
        :return: tensor represent for predicted y
        """
        with tf.name_scope('word_embed'):
            # Look up embeddings for inputs.
            self._word_embeddings = tf.Variable(
                tf.random_uniform(
                    [self._vocabulary_size, self.embedding_size],
                    -1.0,
                    1.0), name='word_embedding')

        with tf.name_scope('lookup'):
            embed = tf.nn.embedding_lookup(self._word_embeddings, x)

        return embed

    def loss(self, predicted_y, y):
        reshaped_y = tf.reshape(
            tf.cast(y, dtype=tf.int64),
            [-1, 1]
        )

        with tf.name_scope('nce_variable'):
            # Construct the variables for the NCE loss
            self._nce_weights = tf.Variable(
                tf.truncated_normal(
                    [self._vocabulary_size, self.embedding_size],
                    stddev=1.0 / math.sqrt(self.embedding_size)
                ), name='nce_weights'
            )
            nce_biases = tf.Variable(tf.zeros([self._vocabulary_size]),
                                     name='nce_biases')

        loss_op = tf.reduce_mean(
            tf.nn.nce_loss(weights=self._nce_weights,
                           biases=nce_biases,
                           labels=reshaped_y,
                           inputs=predicted_y,
                           num_sampled=self.num_sampled,
                           num_classes=self._vocabulary_size)
        )

        tf.summary.scalar("loss", loss_op)

        self.loss_op = loss_op

        return loss_op

    @staticmethod
    def learning_rate(initial_learning_rate=1.0,
                      decay_steps=100000,
                      learning_rate_decay_factor=0.99):
        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            tf.train.get_global_step(),
            decay_steps,
            learning_rate_decay_factor,
            staircase=True
        )

        tf.summary.scalar('learning_rate', learning_rate)

        return learning_rate

    def optimize(self, loss_op, learning_rate):
        # Compute gradients.
        with tf.control_dependencies([loss_op]):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss_op)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + '/histogram', var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step(),
                                      name='optimizer')

        return train_op

    def train(self, training_input, labels, log_dir,
              epoch_per_step,
              initial_learning_rate=1.0, decay_steps=1000000,
              learning_rate_decay_factor=0.99):
        predicted_label = self.inference(training_input)

        # Calculate loss.
        loss_op = self.loss(predicted_label, labels)

        learning_rate = self.learning_rate(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            learning_rate_decay_factor=learning_rate_decay_factor
        )
        train_op = self.optimize(loss_op, learning_rate)

        self.write_metadata(log_dir)

        training_epoch = tf.multiply(
            tf.cast(epoch_per_step, tf.float64),
            tf.cast(tf.train.get_global_step(), tf.float64)
        )
        tf.summary.scalar('training_epoch', training_epoch)

        return train_op

    @staticmethod
    def export_as_gensim_word2vec_format(graph, sess, output_file, normalize=False):
        word_embeddings_tensor = graph.get_tensor_by_name("word_embed/word_embedding:0")

        if not normalize:
            word_embedding = sess.run(word_embeddings_tensor)
        else:
            word_embedding = sess.run(
                tf.nn.l2_normalize(word_embeddings_tensor, dim=1)
            )

        word_array = sess.run(graph.get_tensor_by_name("word_array:0"))
        vocabulary_size = sess.run(graph.get_tensor_by_name("vocabulary_size:0"))
        embedding_size = sess.run(graph.get_tensor_by_name("embedding_size:0"))

        word_array_string = np.vectorize(lambda x: x.decode("utf-8"))(word_array)
        word_matrix = np.c_[word_array_string, word_embedding].tolist()

        word2vec_str = '\n'.join(map(lambda x: " ".join(x), word_matrix))

        with open(output_file, 'wt') as fd:
            fd.write(
                "{} {}\n".format(vocabulary_size, embedding_size)
            )
            fd.write(word2vec_str)

    def write_metadata(self, log_dir):
        # TODO: if session is loaded from ckpt, then is should not execute
        log_dir_abs_path = os.path.realpath(log_dir)

        summary_writer = tf.summary.FileWriter(log_dir)

        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = self._word_embeddings.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(log_dir_abs_path,
                                               'metadata.tsv')

        # normalized_embedding = config.embeddings.add()
        # normalized_embedding.tensor_name = self._normalized_embeddings.name
        # # Link this tensor to its metadata file (e.g. labels).
        # normalized_embedding.metadata_path = os.path.join(
        #     log_dir_abs_path,
        #     'metadata.tsv')

        nce_weight = config.embeddings.add()
        nce_weight.tensor_name = self._nce_weights.name
        # Link this tensor to its metadata file (e.g. labels).
        nce_weight.metadata_path = os.path.join(log_dir_abs_path,
                                                'metadata.tsv')

        projector.visualize_embeddings(summary_writer, config)

        # output metadata of embedding vector
        with open(os.path.join(log_dir_abs_path, 'metadata.tsv'), 'wt') as fd:
            # output body
            fd.write('\n'.join(map(lambda x: self._index_to_word_map[x],
                                   range(self._vocabulary_size))))

        summary_writer.close()