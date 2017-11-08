#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .BaseRecordParser import BaseRecordParser


class CSVBasedRecordParser(BaseRecordParser):
    def __init__(self, record_defaults=tuple([[0], [0]])):

        self.record_defaults = record_defaults

        super(CSVBasedRecordParser, self).__init__()

    def parse_function(self, record_string):
        current_word, context_word = tf.decode_csv(
            record_string,
            record_defaults=self.record_defaults
        )
        return current_word, context_word
