try:
    import cPickle as pickle
except ImportError:
    import pickle

from .BaseMetadataLoader import BaseMetadataLoader


class PickleBasedMetadataLoader(BaseMetadataLoader):
    def get_metadata(self, metadata_file, window_based_metadata_file):
        with open(metadata_file, 'rb') as fd:
            data = pickle.load(fd)

        with open(window_based_metadata_file, 'rb') as fd:
            window_based_data = pickle.load(fd)

        return data['vocabulary_size'], data['index2id'], data['vocabulary_count'], window_based_data['epoch_size']
