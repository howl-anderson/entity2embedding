import tensorflow as tf

from .BaseCorporaLoader import BaseCorporaLoader
from .BaseRecordParser import BaseRecordParser
from .BaseMetadataLoader import BaseMetadataLoader


class DatasetBasedCorporaLoader(BaseCorporaLoader):
    def __init__(self,
                 corpora_file_list,
                 metadata_file,
                 window_based_metedata_file,
                 dataset_loader,
                 record_parser,
                 metadata_loader):
        # type: (list, str, str, tf.contrib.data.Dataset, BaseRecordParser, BaseMetadataLoader) -> None

        self.corpora_file_list = corpora_file_list
        self.metadata_file = metadata_file
        self.window_based_metedata_file = window_based_metedata_file
        self.dataset_loader = dataset_loader
        self.record_parser = record_parser
        self.metadata_loader = metadata_loader

        self.buffer_size = 10000

        self._iterator = None
        self._dataset = None
        self._batched_dataset = None

        super(DatasetBasedCorporaLoader, self).__init__()

    def load(self):
        dataset = self.dataset_loader(self.corpora_file_list)

        # Parse the record into tensors.
        dataset = dataset.map(self.record_parser.parse_function)

        # Repeat the input indefinitely.
        dataset = dataset.repeat()

        # Randomly shuffling data
        dataset = dataset.shuffle(buffer_size=self.buffer_size)

        self._dataset = dataset

    def set_batch_size(self, batch_size):
        self._batched_dataset = self._dataset.batch(batch_size)
        self._iterator = self._batched_dataset.make_initializable_iterator()
        # self._iterator = self._batched_dataset.make_one_shot_iterator()

    def get_batch(self, batch_size):
        self.load()
        self.set_batch_size(batch_size)
        return self._iterator.get_next()

    def initializer(self):
        return self._iterator.initializer

    def get_metadata(self):
        return self.metadata_loader.get_metadata(self.metadata_file, self.window_based_metedata_file)
