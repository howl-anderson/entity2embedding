#!/usr/bin/python
# -*- coding: utf-8 -*-
import functools
from multiprocessing.pool import ThreadPool

from typing import (
    Any
)

from .BaseCorporaBuilder import BaseCorporaBuilder
from .BaseLineProcessor import BaseLineProcessor
from .BaseDataExporter import BaseDataExporter


class LineBasedCorporaBuilder(BaseCorporaBuilder):
    def __init__(self,
                 corpora_file,
                 line_processor,
                 data_exporter):
        # type: (str, BaseLineProcessor, BaseDataExporter) -> None

        self.corpora_file = corpora_file
        self.line_processor = line_processor
        self.data_exporter = data_exporter

        self._data = []

        super(LineBasedCorporaBuilder, self).__init__()

    def do_line_process(self, record_string):
        # type: (str) -> Any
        return self.line_processor.line_process(record_string)

    def build(self):
        # type: () -> int
        with open(self.corpora_file, 'rt') as fd:
            line_result_list = functools.reduce(
                lambda x, y: x + y,
                map(self.do_line_process, fd)
            )
            self._data.extend(line_result_list)

        return len(self._data)

    def export(self, output_file):
        self.data_exporter.export(self._data, output_file)
