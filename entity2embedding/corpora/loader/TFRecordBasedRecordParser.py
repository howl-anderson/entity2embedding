#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .BaseRecordParser import BaseRecordParser


class TFRecordBasedRecordParser(BaseRecordParser):
    def parse_function(self, example_proto):
        features = {
            "target": tf.FixedLenFeature((), tf.int64, default_value=0),
            "context": tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["target"], parsed_features["context"]
