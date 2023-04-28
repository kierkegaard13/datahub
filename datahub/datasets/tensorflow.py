import os
import shutil
from typing import Optional, Any

from datasets.dataset import Dataset

import tensorflow_datasets as tfds


class TFDataset(Dataset):
    def __init__(self, name: str, download_directory: str = "data/"):
        self.name = name
        self.download_directory = download_directory

    @staticmethod
    def list():
        return tfds.list_builders()

    @classmethod
    def get(cls, name, download_directory="data/"):
        if name not in cls.list():
            raise ValueError("Invalid dataset name. Try calling list() to get a list of valid names.")
        return cls(name, download_directory)

    def download(
        self,
        builder_kwargs: Optional[dict[str, Any]] = None,
        download_and_prepare_kwargs: Optional[dict[str, Any]] = None
    ):
        if builder_kwargs is None:
            builder_kwargs = {}
        if download_and_prepare_kwargs is None:
            download_and_prepare_kwargs = {}
        self.builder = tfds.builder(self.name, data_dir=self.download_directory, **builder_kwargs)
        self.builder.download_and_prepare(**download_and_prepare_kwargs)

    def clean(self):
        shutil.rmtree(os.path.join(self.download_directory, "downloads"))
        shutil.rmtree(os.path.join(self.download_directory, self.name))

    def read(
        self,
        split=None,
        batch_size=None,
        as_supervised: bool = False,
        read_config=None,
        decoders=None,
        shuffle: bool = False,
        to_numpy: bool = False,
        as_dataset_kwargs: Optional[dict[str, Any]] = None
    ):
        if not hasattr(self, "builder"):
            raise AttributeError("Must call download before read")
        if as_dataset_kwargs is None:
            as_dataset_kwargs = {}
        ds = self.builder.as_dataset(
            split=split,
            as_supervised=as_supervised,
            shuffle_files=shuffle,
            read_config=read_config,
            batch_size=batch_size,
            decoders=decoders,
            **as_dataset_kwargs
        )
        if to_numpy:
            tfds.as_numpy(ds)
        return ds
