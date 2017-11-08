#!/usr/bin/python
# -*- coding: utf-8 -*-


class BaseRecordParser(object):
    def __init__(self, *args, **kwargs):
        pass

    def parse_function(self, record_string):
        raise NotImplementedError
