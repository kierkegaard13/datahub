from typing import Optional, Any

from datasets.dataset import Dataset

import tensorflow_datasets as tfds


class TFDataset(Dataset):
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def list():
        return tfds.list_builders()

    @classmethod
    def get(cls, name):
        if name not in cls.list():
            raise ValueError("Invalid dataset name. Try calling list() to get a list of valid names.")
        return cls(name)

    def download(
        self,
        directory: str = "data/",
        builder_kwargs: Optional[dict[str, Any]] = None,
        download_and_prepare_kwargs: Optional[dict[str, Any]] = None
    ):
        if builder_kwargs is None:
            builder_kwargs = {}
        if download_and_prepare_kwargs is None:
            download_and_prepare_kwargs = {}
        self.builder = tfds.builder(self.name, data_dir=directory, **builder_kwargs)
        self.builder.download_and_prepare(**download_and_prepare_kwargs)

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
            decoders=decoders,
            **as_dataset_kwargs
        )
        if to_numpy:
            tfds.as_numpy(ds)
        return ds
