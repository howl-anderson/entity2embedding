import collections
import errno
import os

from typing import List


def list_unhidden_file_in_dir(input_dir: str) -> List[str]:
    all_nodes = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    all_files = filter(os.path.isfile, all_nodes)
    all_unhidden_files = filter(
        lambda x: not os.path.basename(x).startswith("."), all_files
    )
    return list(all_unhidden_files)


def make_dir_recursive(dir_path: str):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def dict_to_namedtuple(d, name="Dict2Namedtuple"):
    return collections.namedtuple(name, d.keys())(*d.values())
