from abc import ABC, abstractmethod


class Dataset(ABC):
    registry: dict[str, "Dataset"] = {}

    def __init_subclass__(cls):
        if hasattr(cls, "name"):
            assert cls.name not in Dataset.registry, f"Dataset: {cls.name} is already registered"
            Dataset.registry[cls.name] = cls

    def list_datasets(self):
        for key in self.registry:
            print(key)

    @abstractmethod
    def download(self, directory: str = "data/") -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, from_disk: bool = True):
        raise NotImplementedError
