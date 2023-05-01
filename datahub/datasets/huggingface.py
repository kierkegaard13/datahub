from datasets import load_dataset, list_datasets

from datahub.datasets.dataset import Dataset


class HFDataset(Dataset):
    def __init__(self, method: str):
        self.method = method

    @staticmethod
    def list(**list_kwargs):
        return list_datasets(**list_kwargs)

    def download(self):
        pass

    @staticmethod
    def read(**read_kwargs):
        return load_dataset(**read_kwargs)
