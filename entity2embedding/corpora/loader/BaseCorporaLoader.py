#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import (
    Tuple, Dict
)

import tensorflow as tf


class BaseCorporaLoader(object):
    def __init__(self, *args, **kwargs):
        pass

    def load(self):
        pass

    def set_batch_size(self, batch_size):
        pass

    def initializer(self):
        return tf.no_op()

    def get_batch(self, batch_size):
        raise NotImplementedError

    def get_metadata(self):
        # type: () -> Tuple[int, Dict[int, str], Dict[int, object]]
        raise NotImplementedError
