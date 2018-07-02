#!/usr/bin/python
# -*- coding: utf-8 -*-

import warnings

from typing import (
    Any
)

from .BaseLineProcessor import BaseLineProcessor
from .window_frame_extract import window_frame_extract


class ContextWindowBasedLineProcessor(BaseLineProcessor):
    def __init__(self, skip_window=3, record_delimiter=' '):
        self.skip_window = skip_window
        self.record_delimiter = record_delimiter

        super(ContextWindowBasedLineProcessor, self).__init__()

    def line_process(self, record_string):
        # type: (str) -> Any
        record_string = record_string.strip()

        if not len(record_string):
            warnings.warn("Empty line", UserWarning)
            return []

        record_list = record_string.split(self.record_delimiter)

        record_list = list(map(lambda x: int(x), record_list))

        return window_frame_extract(record_list, self.skip_window)

