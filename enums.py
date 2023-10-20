from enum import Enum


class TaskMode(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class ClassMode(Enum):
    STANDARD = 0            # Standard 6-class model
    COMPRESSED_START = 1    # 5-class model which merges classes 1-2 from the original
    COMPRESSED_END = 2      # 5-class model which merges classes 4-5 from the original
    COMPRESSED_BOTH = 3     # 4-class model which merges classes 1-2 and 4-5 from the original
