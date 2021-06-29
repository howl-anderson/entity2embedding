#!/usr/bin/python
# -*- coding: utf-8 -*-
import functools
import os
import multiprocessing
import pickle

try:
    from itertools import izip_longest as zip_longest
except ImportError:
    from itertools import zip_longest

from typing import List, AnyStr

from entity2embedding.corpora.builder.LineBasedCorporaBuilder import (
    LineBasedCorporaBuilder,
)
from entity2embedding.corpora.builder.ContextWindowBasedLineProcessor import (
    ContextWindowBasedLineProcessor,
)
from entity2embedding.corpora.builder.TFRecordExporter import TFRecordExporter

CURRENT_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))


def build_worker_func_wrapper(args):
    instance = args[0]
    return instance._build_worker(args[1])


class LineBasedCorporaBatchBuilder(object):
    """
    batch executor for map each file onto LineBasedCorporaBuilder
    """

    def __init__(
        self,
        corpora_file_list,
        output_dir,
        metadata_file,
        skip_window=3,
        record_delimiter=",",
    ):
        # type: (List, AnyStr, AnyStr, long, AnyStr) -> None

        self.corpora_file_list = corpora_file_list
        self.metadata_output_file = metadata_file
        self.line_processor = ContextWindowBasedLineProcessor(
            skip_window=skip_window, record_delimiter=record_delimiter
        )
        self.data_exporter = TFRecordExporter()

        self.output_dir = output_dir

        self.epoch_size = 0

        super(LineBasedCorporaBatchBuilder, self).__init__()

    def build(self):
        """
        the only execute entry point for this builder
        :return:
        """
        build_results = list(map(self._build_worker, self.corpora_file_list))
        # FIXME: multi-process version has fatal bug in python3
        # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        # build_results = list(pool.imap_unordered(
        #     build_worker_func_wrapper,
        #     zip_longest([self], self.corpora_file_list, fillvalue=self)))

        self.epoch_size = functools.reduce(lambda x, y: x + y, build_results)

        self._write_metadata()

    def _build_worker(self, corpora_file):
        """
        build partial corpus for an input file
        :param corpora_file: string, input corpora file
        :return:
        """
        output_file = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(corpora_file))[0] + ".tfrecord",
        )
        print(output_file)
        corpora = LineBasedCorporaBuilder(
            corpora_file, self.line_processor, self.data_exporter
        )
        corpora_pair_size = corpora.build()
        corpora.export(output_file)
        return corpora_pair_size

    def _write_metadata(self):
        pickle_data = {"epoch_size": self.epoch_size}

        with open(self.metadata_output_file, "wb") as fd:
            pickle.dump(pickle_data, fd)
