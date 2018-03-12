import json
import os

from entity2embedding.utils import make_dir_recursive


def load_project_structure(project_dir, create=True):
    dirs_need_be_created = {
        'RAW_DATA_DIR': 'data/raw_data',
        'INTERMEDIATE_DATA_DIR': 'data/intermediate_data',
        'FINAL_DATA_DIR': 'data/final_data',
        'CORPORA_DATA_DIR': 'data/final_data/corpora',
        'LOG_DIR': 'log'
    }

    project_struct = {}

    for k, i in dirs_need_be_created.items():
        full_path = os.path.join(project_dir, i)
        make_dir_recursive(full_path)
        project_struct[k] = full_path

    project_struct['ONE_HOT_METADATA_FILE'] = os.path.join(
        project_dir,
        'data/final_data/meta.pkl')
    project_struct['WINDOW_BASED_METADATA_FILE'] = os.path.join(
        project_dir,
        'data/final_data/window_based_metadata.pkl')
    project_struct['GENSIM_EXPORT_FILE'] = os.path.join(
        project_dir,
        'gensim_compatible_word2vec.txt')

    return project_struct


def build_config(config_file=None, cmdline_args=None):
    config = {}

    if config_file is not None:
        with open(config_file, "rt") as fd:
            file_config = json.load(fd)
            config.update(file_config)

    if cmdline_args is not None:
        config.update(cmdline_args)

    return config
