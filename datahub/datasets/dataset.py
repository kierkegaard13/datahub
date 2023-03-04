from abc import ABC, abstractmethod


class Dataset(ABC):
    registry: dict[str, "Dataset"] = {}

    def __init_subclass__(cls, name):
        assert name not in Dataset.registry, f"Dataset: {name} is already registered"
        Dataset.registry[name] = cls

    def list_datasets(self):
        for key in self.registry:
            print(key)

    @abstractmethod
    def download(self):
        raise NotImplementedError

    @abstractmethod
    def read(self, from_disk: bool = True):
        raise NotImplementedError
