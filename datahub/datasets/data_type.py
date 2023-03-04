from enum import Enum, auto


class DataType(Enum):
    TABULAR = auto()
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    MULTI = auto()
    TIMESERIES = auto()
    GRAPH = auto()
