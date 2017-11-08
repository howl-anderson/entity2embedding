#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from typing import (
    AnyStr, List
)

from .BaseLineProcessor import BaseLineProcessor


class TargetContextBasedLineProcessor(BaseLineProcessor):
    def line_process(self, record_str):
        # type: (str) -> List
        record_list = record_str.split(sep=',')
        if len(record_list) != 2:
            raise ValueError("Length of record_list: {} is not 2".format(record_list))
        current_word, context_word = map(lambda x: int(x), record_list)
        return [(current_word, context_word)]
