# -*- coding: utf-8 -*-

import tensorflow as tf

from .BaseDataExporter import BaseDataExporter


class TFRecordExporter(BaseDataExporter):
    def export(self, data, output_file):
        def translate_func(pair):
            target, context = pair
            example = tf.train.Example(features=tf.train.Features(feature={
                "target": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[target])),
                'context': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[context]))
            }))
            return example.SerializeToString()  # serial to string

        record_string_iterator = map(translate_func, data)

        with tf.io.TFRecordWriter(output_file) as writer:
            list(map(writer.write, record_string_iterator))
