#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .DatasetBasedCorporaLoader import DatasetBasedCorporaLoader
from .TFRecordBasedRecordParser import TFRecordBasedRecordParser
from .PickleBasedMetadataLoader import PickleBasedMetadataLoader


class TFRecordBasedCorporaLoader(DatasetBasedCorporaLoader):
    def __init__(self, corpora_file_list, metadata_file, window_based_metadata_file):
        # type: (list, str, str) -> None
        dataset_loader = tf.data.TFRecordDataset
        record_parser = TFRecordBasedRecordParser()
        metadata_loader = PickleBasedMetadataLoader()

        super(TFRecordBasedCorporaLoader, self).__init__(
            corpora_file_list,
            metadata_file,
            window_based_metadata_file,
            dataset_loader,
            record_parser,
            metadata_loader,
        )
