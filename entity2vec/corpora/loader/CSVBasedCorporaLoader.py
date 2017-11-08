#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .PickleBasedMetadataLoader import PickleBasedMetadataLoader
from .DatasetBasedCorporaLoader import DatasetBasedCorporaLoader
from .CSVBasedRecordParser import CSVBasedRecordParser


class CSVBasedCorporaLoader(DatasetBasedCorporaLoader):
    def __init__(self, corpora_file_list, metadata_file):
        # type: (list, str) -> None

        dataset_loader = tf.contrib.data.TextLineDataset
        record_parser = CSVBasedRecordParser()
        metadata_loader = PickleBasedMetadataLoader()

        super(CSVBasedCorporaLoader, self).__init__(corpora_file_list, metadata_file, dataset_loader, record_parser, metadata_loader)
