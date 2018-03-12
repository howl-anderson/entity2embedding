#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import warnings

from typing import (
    Any
)

from .BaseLineProcessor import BaseLineProcessor


class ContextWindowBasedLineProcessor(BaseLineProcessor):
    def __init__(self, skip_window=3, record_delimiter=' '):
        self.skip_window = skip_window
        self.record_delimiter = record_delimiter

        super(ContextWindowBasedLineProcessor, self).__init__()

    def line_process(self, record_string):
        # type: (str) -> Any
        target_buffer = []
        context_buffer = []

        record_string = record_string.strip()

        if not len(record_string):
            warnings.warn("Empty line", UserWarning)
            return zip(target_buffer, context_buffer)

        record_list = record_string.split(self.record_delimiter)

        record_list = list(map(lambda x: int(x), record_list))
        padding = [None] * self.skip_window
        line_with_padding = padding + record_list + padding

        # window => [ skip_window target skip_window ]
        window_span = 2 * self.skip_window + 1

        window = collections.deque(maxlen=window_span)

        # fulfill the window
        window.extend(line_with_padding[:window_span])
        word_index = window_span

        center_index_of_window = self.skip_window

        while True:
            context_range = list(range(0, self.skip_window)) \
                          + list(range(self.skip_window + 1, window_span))
            for i in context_range:
                context_word = window[i]

                # skip None
                if context_word is None:
                    continue

                target_buffer.append(window[center_index_of_window])
                context_buffer.append(context_word)

            # update the window or exit
            try:
                next_word = line_with_padding[word_index]
            except IndexError:
                # reach the EOL
                break
            else:
                window.append(next_word)
                word_index += 1

        return list(zip(target_buffer, context_buffer))
