import collections
import shutil
import tempfile

import os
import errno
import time
from datetime import datetime

import tensorflow as tf

from entity2embedding.corpora.builder.LineBasedCorporaBatchBuilder import \
    LineBasedCorporaBatchBuilder
from entity2embedding.corpora.loader import TFRecordBasedCorporaLoader
from entity2embedding.corpora.preprocessor.TextBasedPreprocessor import \
    TextBasedPreprocessor
from entity2embedding.corpora.loader.CorporaInitializer import CorporaInitializer
from entity2embedding.word2vec.skip_gram.basic import BasicWord2vec


class _DefaultOptionValue(object):
    pass


class Word2Vec(object):
    def __init__(self, sentences=None, size=100, alpha=_DefaultOptionValue(),
                 window=5, min_count=5, max_vocab_size=_DefaultOptionValue(),
                 sample=_DefaultOptionValue(), seed=_DefaultOptionValue(),
                 workers=_DefaultOptionValue(),
                 min_alpha=_DefaultOptionValue(), sg=_DefaultOptionValue(),
                 hs=_DefaultOptionValue(), negative=5,
                 cbow_mean=_DefaultOptionValue(), iter=5, null_words=_DefaultOptionValue(),
                 trim_rule=_DefaultOptionValue(), sorted_vocab=_DefaultOptionValue(),
                 batch_words=10000):
        self.sentences = sentences
        self.size = size  # embedding size
        self.min_count = min_count
        self.window = window
        self.batch_words = batch_words
        self.negative = negative
        self.iter = iter

        #
        self.log_dir = None
        self.log_frequency = 60

        self.workspace_dir = tempfile.mkdtemp(
            prefix="tensorflow_gensim_",
            suffix="_{}".format(datetime.now().isoformat()),
            dir=os.getenv('TF_GENSIM_TMP_DIR', None)
        )

        print("Workspace at: {}".format(self.workspace_dir))

        # build corpora
        self.build_corpora()

        # train
        self.train()

    @staticmethod
    def make_dir_if_not_exists(input_dir):
        try:
            os.makedirs(input_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def list_all_files_in_dir(input_dir):
        result = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                result.append(os.path.join(input_dir, f))

        return result

    def build_corpora(self):
        # make workspace directory and copy data

        input_dir = os.path.join(self.workspace_dir, 'raw_data')
        output_dir = os.path.join(self.workspace_dir, 'final_data')
        output_data_dir = os.path.join(self.workspace_dir, 'final_data/data')
        output_corpora_dir = os.path.join(self.workspace_dir, 'final_data/corpora')

        self.make_dir_if_not_exists(input_dir)
        self.make_dir_if_not_exists(output_dir)
        self.make_dir_if_not_exists(output_data_dir)
        self.make_dir_if_not_exists(output_corpora_dir)

        shutil.copy(self.sentences.source, input_dir)

        # preprocess data
        input_files = self.list_all_files_in_dir(input_dir)

        processor = TextBasedPreprocessor(input_files, output_dir=output_data_dir,
                                          min_count=self.min_count,
                                          metadata_file=os.path.join(output_dir, 'meta.pkl')
                                          )
        processor.build()

        # build corpora data
        input_files = self.list_all_files_in_dir(output_data_dir)

        corpora = LineBasedCorporaBatchBuilder(input_files, output_corpora_dir,
                                               skip_window=self.window,
                                               metadata_file=os.path.join(output_dir, 'window_based_metadata.pkl'))
        corpora.build()

    def train(self):
        self.log_dir = os.path.join(self.workspace_dir, 'log')
        self.make_dir_if_not_exists(self.log_dir)

        input_dir = os.path.join(self.workspace_dir, 'final_data/corpora')
        metadata_file = os.path.join(self.workspace_dir, 'final_data/meta.pkl')
        window_based_metadata_file = os.path.join(self.workspace_dir, 'final_data/window_based_metadata.pkl')
        input_files = self.list_all_files_in_dir(input_dir)

        corpora_object = TFRecordBasedCorporaLoader.TFRecordBasedCorporaLoader(
            input_files, metadata_file, window_based_metadata_file)

        other = self

        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                training_input, labels = corpora_object.get_batch(
                    self.batch_words)

                vocabulary_size, index_to_word_map, vocabulary_count, epoch_size = corpora_object.get_metadata()

                max_steps = int(self.iter * (epoch_size / self.size))

                word2vec = BasicWord2vec(vocabulary_size, index_to_word_map,
                                         self.size,
                                         self.negative)

                train_op = word2vec.train(training_input, labels,
                                          self.log_dir,
                                          epoch_per_step=epoch_size / self.batch_words,
                                          initial_learning_rate=1.0,
                                          decay_steps=1000000,
                                          learning_rate_decay_factor=0.99
                                          )

                loss_op = word2vec.loss_op

                class _LoggerHook(tf.train.SessionRunHook):
                    """Logs loss and runtime."""

                    def begin(self):
                        self._step = -1
                        self._start_time = time.time()

                    def before_run(self, run_context):
                        self._step += 1
                        return tf.train.SessionRunArgs(
                            loss_op)  # Asks for loss value.

                    def after_run(self, run_context, run_values):
                        if self._step % other.log_frequency == 0:
                            current_time = time.time()
                            duration = current_time - self._start_time
                            self._start_time = current_time

                            loss_value = run_values.results
                            examples_per_sec = other.log_frequency * other.batch_words / duration
                            sec_per_batch = float(
                                duration / other.log_frequency)

                            format_str = (
                                '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                            print(format_str % (
                            datetime.now(), self._step, loss_value,
                            examples_per_sec, sec_per_batch))

                with tf.train.MonitoredTrainingSession(
                        checkpoint_dir=self.log_dir,
                        save_summaries_steps=None,
                        save_summaries_secs=60,
                        save_checkpoint_secs=60,
                        hooks=[
                            CorporaInitializer(corpora_object),
                            tf.train.StopAtStepHook(last_step=max_steps),
                            tf.train.NanTensorHook(loss_op),
                            _LoggerHook(),
                        ]
                ) as mon_sess:
                    while not mon_sess.should_stop():
                        mon_sess.run(train_op)

    @property
    def wv(self):
        return self

    def save_word2vec_format(self, fname, fvocab=_DefaultOptionValue(),
                             binary=_DefaultOptionValue(),
                             total_vec=_DefaultOptionValue()):
        latest_checkpoint = tf.train.latest_checkpoint(self.log_dir)

        with tf.Session() as sess:
            meta_graph_file = '.'.join([latest_checkpoint, "meta"])
            saver = tf.train.import_meta_graph(meta_graph_file)

            saver.restore(sess, latest_checkpoint)

            BasicWord2vec.export_as_gensim_word2vec_format(sess.graph, sess, fname)


class LineSentence(object):
    def __init__(self, source, max_sentence_length=_DefaultOptionValue(), limit=_DefaultOptionValue()):
        # waring about incompatible feature
        if not isinstance(max_sentence_length, _DefaultOptionValue):
            raise ValueError(
                "parameter `max_sentence_length` is not supported."
            )
        if not isinstance(limit, _DefaultOptionValue):
            raise ValueError(
                "parameter `limit` is not supported."
            )

        self.source = source
