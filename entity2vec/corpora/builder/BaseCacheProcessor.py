#!/usr/bin/python
# -*- coding: utf-8 -*-


class BaseCacheProcessor(object):
    def __init__(self, *args, **kwargs):
        pass

    def get_cache(self, ):
        raise NotImplementedError

    def set_cache(self, data):
        raise NotImplementedError
