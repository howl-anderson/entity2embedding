from typing import (
    Tuple, Dict
)


class BaseMetadataLoader(object):
    def get_metadata(self, metadata_file, window_based_metadata_file):
        # type: (...) -> Tuple[int, Dict[int, str], Dict[int, object], int]
        raise NotImplementedError
