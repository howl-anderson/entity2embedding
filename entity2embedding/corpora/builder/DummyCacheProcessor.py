#!/usr/bin/python
# -*- coding: utf-8 -*-

from .BaseCacheProcessor import BaseCacheProcessor


class DummyCacheProcessor(BaseCacheProcessor):
    def get_cache(self, ):
        raise NotImplementedError

    def set_cache(self, data):
        raise NotImplementedError
