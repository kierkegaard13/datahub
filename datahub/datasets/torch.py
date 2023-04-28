import os
import shutil

import torchtext.datasets as text_datasets
import torchvision.datasets as vision_datasets
import torchaudio.datasets as audio_datasets
from torch.utils.data import DataLoader

from datahub.datasets.dataset import Dataset


def get_datasets(attribute_list):
    return [item for item in attribute_list if item[0].isupper()]


class TorchDataset(Dataset):
    library: str = None

    def __init__(self, name: str, download_directory: str = "data/"):
        self.name = name
        self.download_directory = download_directory

    @classmethod
    def list(cls):
        return get_datasets(dir(cls.library))

    @classmethod
    def get(cls, name, download_directory="data/"):
        if name not in cls.list():
            raise ValueError("Invalid dataset name. Try calling list() to get a list of valid names.")
        return cls(name, download_directory)

    def clean(self):
        shutil.rmtree(os.path.join(self.download_directory, self.name))


class TorchTextDataset(TorchDataset):
    library = text_datasets

    def read(self, **read_kwargs):
        return getattr(self.library, self.name)(self.download_directory, **read_kwargs)


class TorchVisionDataset(TorchDataset):
    library = vision_datasets

    def download(self, **download_kwargs):
        self.ds = getattr(self.library, self.name)(root=self.download_directory, download=True, **download_kwargs)

    def read(self, batch_size: int, shuffle: bool, **loader_kwargs):
        return DataLoader(self.ds, batch_size, shuffle, **loader_kwargs)


class TorchAudioDataset(TorchDataset):
    library = audio_datasets

    def download(self, **download_kwargs):
        try:
            self.ds = getattr(self.library, self.name)(root=self.download_directory, download=True, **download_kwargs)
        except TypeError:
            self.ds = getattr(self.library, self.name)(root=self.download_directory, **download_kwargs)

    def read(self, batch_size: int, shuffle: bool, **loader_kwargs):
        return DataLoader(self.ds, batch_size, shuffle, **loader_kwargs)
