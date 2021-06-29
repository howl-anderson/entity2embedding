import time
from datetime import datetime
import argparse
import pprint

import tensorflow as tf

from entity2embedding.corpora.loader.CorporaInitializer import CorporaInitializer
from entity2embedding.corpora.loader import TFRecordBasedCorporaLoader
from entity2embedding.word2vec.skip_gram.basic import BasicWord2vec
from entity2embedding.shortcuts.utils import load_project_structure, build_config
from entity2embedding.utils import list_unhidden_file_in_dir, dict_to_namedtuple


def create_argparser():
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument(
        "-p", "--project", help="Directory where to write project files."
    )
    parser.add_argument(
        "-m", "--max_steps", default=-1, help="Number of batches to run."
    )
    parser.add_argument(
        "-d",
        "--log_device_placement",
        default=False,
        help="Whether to log device placement.",
    )
    parser.add_argument(
        "-f",
        "--log_frequency",
        default=10,
        help="How often to log results to the console.",
    )
    parser.add_argument(
        "-b", "--batch_size", default=128, help="How much one batch holds."
    )
    parser.add_argument(
        "-e", "--embedding_size", default=128, help="How much embedding vector size."
    )
    parser.add_argument(
        "-n", "--num_sampled", default=64, help="How much negative samples are used."
    )
    return parser


def train(corpora_object, log_dir, FLAGS):
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            training_input, labels = corpora_object.get_batch(FLAGS.batch_size)
            import pdb; pdb.set_trace()

            vocabulary_size, index_to_word_map, vocabulary_count, epoch_size = (
                corpora_object.get_metadata()
            )

            word2vec = BasicWord2vec(
                vocabulary_size,
                index_to_word_map,
                FLAGS.embedding_size,
                FLAGS.num_sampled,
            )

            train_op = word2vec.train(
                training_input,
                labels,
                log_dir,
                FLAGS.batch_size / epoch_size,
                initial_learning_rate=1.0,
                decay_steps=1000000,
                learning_rate_decay_factor=0.99,
            )

            loss_op = word2vec.loss_op

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""

                def __init__(self):
                    self._step = None
                    self._start_time = None

                    super(_LoggerHook, self).__init__()

                def begin(self):
                    self._step = -1
                    self._start_time = time.time()

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs(loss_op)  # Asks for loss value.

                def after_run(self, run_context, run_values):
                    if self._step % FLAGS.log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time

                        loss_value = run_values.results
                        examples_per_sec = (
                            FLAGS.log_frequency * FLAGS.batch_size / duration
                        )
                        sec_per_batch = float(duration / FLAGS.log_frequency)

                        format_str = (
                            "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f "
                            "sec/batch)"
                        )
                        print(
                            format_str
                            % (
                                datetime.now(),
                                self._step,
                                loss_value,
                                examples_per_sec,
                                sec_per_batch,
                            )
                        )

            with tf.train.MonitoredTrainingSession(
                checkpoint_dir=log_dir,
                save_summaries_steps=None,
                save_summaries_secs=60,
                save_checkpoint_secs=60,
                hooks=[
                    CorporaInitializer(corpora_object),
                    # tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                    tf.train.NanTensorHook(loss_op),
                    _LoggerHook(),
                ],
                config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
            ) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)


if __name__ == "__main__":
    arg_parser = create_argparser()
    cmdline_args = {
        k: v for k, v in list(vars(arg_parser.parse_args()).items()) if v is not None
    }

    config = build_config(cmdline_args.get("config"), cmdline_args)

    pprint.pprint(config)

    project_dir = config.get("project")

    project = load_project_structure(project_dir)

    input_files = list_unhidden_file_in_dir(project["CORPORA_DATA_DIR"])

    corpora_object = TFRecordBasedCorporaLoader.TFRecordBasedCorporaLoader(
        input_files,
        project["ONE_HOT_METADATA_FILE"],
        project["WINDOW_BASED_METADATA_FILE"],
    )

    train(corpora_object, project["LOG_DIR"], dict_to_namedtuple(config))
