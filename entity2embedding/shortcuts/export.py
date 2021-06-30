from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint

import tensorflow as tf

from entity2embedding.shortcuts.utils import load_project_structure, build_config
from entity2embedding.word2vec.skip_gram.basic import BasicWord2vec


def create_argparser():
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument(
        "-p", "--project", help="Directory where to write project files."
    )

    return parser


def export(log_dir, output_file):
    latest_checkpoint = tf.train.latest_checkpoint(log_dir)

    with tf.Session() as sess:
        meta_graph_file = ".".join([latest_checkpoint, "meta"])
        saver = tf.train.import_meta_graph(meta_graph_file)

        saver.restore(sess, latest_checkpoint)

        BasicWord2vec.export_as_gensim_word2vec_format(sess.graph, sess, output_file)


if __name__ == "__main__":
    arg_parser = create_argparser()
    cmdline_args = {
        k: v for k, v in list(vars(arg_parser.parse_args()).items()) if v is not None
    }

    config = build_config(cmdline_args.get("config"), cmdline_args)

    pprint.pprint(config)

    project_dir = config.get("project")

    project = load_project_structure(project_dir)

    export(project["LOG_DIR"], project["GENSIM_EXPORT_FILE"])
