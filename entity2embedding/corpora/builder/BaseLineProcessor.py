#!/usr/bin/python
# -*- coding: utf-8 -*-


class BaseLineProcessor(object):
    def __init__(self, *args, **kwargs):
        pass

    def line_process(self, record_string):
        raise NotImplementedError
