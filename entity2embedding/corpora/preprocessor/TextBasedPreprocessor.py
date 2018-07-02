import collections
import csv
import functools
import os

from joblib import Parallel, delayed

try:
    import cPickle as pickle
except ImportError:
    import pickle


class TextBasedPreprocessor(object):
    """
    1. Turn word-based corpora into index-based (number) corpora,
    2. Word count below min_count will be treat as a special value.
    """
    def __init__(self, corpora_file_list,
                 vocabulary_size=None, output_dir="./output", metadata_file=None, min_count=None):
        # TODO: fix me: metadata_file=None don't work at all!
        if not (vocabulary_size or min_count):
            raise ValueError(
                "Parameter `vocabulary_size` and `min_count`"
                "must have one is not None."
            )

        self.corpora_file_list = corpora_file_list
        self._vocabulary_size = vocabulary_size
        self._min_count = min_count

        self.output_dir = output_dir

        # self.data_output_dir = os.path.join(self.output_dir, "data")
        # if not os.path.exists(self.data_output_dir):
        #     os.makedirs(self.data_output_dir)
        #
        # self.metadata_output_file = os.path.join(self.output_dir, "meta.pkl")

        self.metadata_output_file = metadata_file

        self._index_to_word_map = None

        # mapping word to index, index count from 0: "some-word" => 1
        self._word_to_index_map = dict()

        # mapping index to word_count
        self._index_to_word_count_list = []

    def build(self):
        self._build_word_to_index_map()
        self._build_corpora_data()

        self._index_to_word_map = dict(
            zip(self._word_to_index_map.values(), self._word_to_index_map.keys())
        )

        self._write_metadata()

    def _build_word_to_index_map(self):
        # single-process based worker is replaced by multi-process based solution (see below)
        # counter_iter = map(self._build_word_to_index_map_worker, self.corpora_file_list)
        # counter_list = list(counter_iter)

        counter_list = Parallel(n_jobs=-1)(
            delayed(self._build_word_to_index_map_worker)(i) for i in self.corpora_file_list
        )

        # NOTE: counter + counter is slow, using update maybe better
        def reduce_updater(x, y):
            x.update(y)
            return x
        counter = functools.reduce(reduce_updater, counter_list)

        # add special word 'UNK' stand for unknown words, assign -1 as it's count
        count = [['UNK', -1]]

        # get real vocabulary by settings
        if self._vocabulary_size:
            vocab_counter = counter.most_common(self._vocabulary_size - 1)
        else:
            vocab_counter = list(filter(
                lambda x: x[1] >= self._min_count,
                counter.most_common()
            ))

            # update self._vocabulary_size to correct value
            self._vocabulary_size = len(vocab_counter) + 1

        count.extend(vocab_counter)

        # building mete-data that can map word to index
        for word, word_count in count:
            current_index = len(self._word_to_index_map)
            self._word_to_index_map[word] = current_index
            self._index_to_word_count_list.append(word_count)

    @staticmethod
    def _build_word_to_index_map_worker(corpora_file):
        """
        Turn word string in file to word counter, for get word counter of corpus

        :param corpora_file: string, input file
        :return: word counter
        """
        vocabulary = []
        with open(corpora_file, 'rt') as fd:
            for line in fd:
                clean_line = line.strip()  # remove head and tail empty chars, just in case
                line_word = clean_line.split()
                vocabulary.extend(line_word)

        return collections.Counter(vocabulary)

    def _build_corpora_data(self):
        # single-process based worker is replaced by multi-process based solution (see below)
        # unk_counter_list = list(map(self._build_corpora_data_worker, self.corpora_file_list))

        unk_counter_list = Parallel(n_jobs=-1)(
            delayed(self._build_corpora_data_worker)(i) for i in self.corpora_file_list
        )

        # index 0 means 'UNK'
        self._index_to_word_count_list[0] = sum(unk_counter_list)

    def _build_corpora_data_worker(self, corpora_file):
        output_file = os.path.join(self.output_dir, os.path.basename(corpora_file))

        corpora = []
        with open(corpora_file, 'rt') as fd:
            for line in fd:
                clean_line = line.strip()
                line_word = clean_line.split()
                corpora.append(line_word)

        corpora_data = []
        unk_count = 0
        for line in corpora:
            line_data = list()
            for word in line:
                if word in self._word_to_index_map:
                    index = self._word_to_index_map[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count += 1
                line_data.append(index)
            corpora_data.append(line_data)

        with open(output_file, "w") as fd:
            writer = csv.writer(fd)
            writer.writerows(corpora_data)

        return unk_count

    def _write_metadata(self):
        pickle_data = {
            'vocabulary_size': len(self._index_to_word_map),
            'index2id': self._index_to_word_map,
            'vocabulary_count': self._index_to_word_count_list
        }

        with open(self.metadata_output_file, 'wb') as fd:
            pickle.dump(pickle_data, fd)


def _build_word_to_index_map_worker_func_wrapper(args):
    return TextBasedPreprocessor._build_word_to_index_map_worker(args)
