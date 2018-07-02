import argparse
import os
import pprint
import shutil

from entity2embedding.corpora.builder.LineBasedCorporaBatchBuilder import \
    LineBasedCorporaBatchBuilder
from entity2embedding.corpora.preprocessor.TextBasedPreprocessor import \
    TextBasedPreprocessor
from entity2embedding.shortcuts.utils import load_project_structure, build_config
from entity2embedding.utils import list_unhidden_file_in_dir


def create_argparser():
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument('-c', '--config',
                        help="Config file location")
    parser.add_argument('-p', '--project',
                        help="Directory where to write project files.")
    parser.add_argument('-f', '--source',
                        help="A raw text file or a directory that contains text files")
    parser.add_argument('-s', '--vocabulary_size',
                        help="Vocabulary size")
    parser.add_argument('-w', '--skip_window',
                        help="Skip window Size")
    return parser


if __name__ == "__main__":
    arg_parser = create_argparser()
    cmdline_args = {k: v
                    for k, v in list(vars(arg_parser.parse_args()).items())  # using vars() to turn object attribute to dict
                    if v is not None}

    config = build_config(cmdline_args.get("config"), cmdline_args)

    pprint.pprint(config)

    project_dir = config.get("project")

    project = load_project_structure(project_dir)

    source = config.get("source")
    if not os.path.isfile(source):
        source = list_unhidden_file_in_dir(source)
    else:
        source = [source]

    for f in source:
        shutil.copy(f, project['RAW_DATA_DIR'])

    raw_files = list_unhidden_file_in_dir(project['RAW_DATA_DIR'])

    processor = TextBasedPreprocessor(
        raw_files,
        output_dir=project['INTERMEDIATE_DATA_DIR'],
        metadata_file=project['ONE_HOT_METADATA_FILE'],
        vocabulary_size=config.get('vocabulary_size', 50000))
    processor.build()

    intermediate_files = list_unhidden_file_in_dir(project['INTERMEDIATE_DATA_DIR'])

    corpora = LineBasedCorporaBatchBuilder(
        intermediate_files,
        project['CORPORA_DATA_DIR'],
        project['WINDOW_BASED_METADATA_FILE'],
        skip_window=config.get('skip_window', 1))
    corpora.build()
