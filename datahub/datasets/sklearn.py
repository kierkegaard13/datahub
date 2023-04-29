from sklearn import datasets

from datahub.datasets.dataset import Dataset


class SklearnDataset(Dataset):
    def __init__(self, method: str):
        self.method = method

    @staticmethod
    def list():
        return [item.replace("make_", "").replace("fetch_", "").replace("load_", "") for item in dir(datasets)
                if item.startswith("load") or item.startswith("fetch") or item.startswith("make")]

    @staticmethod
    def list_w_fn():
        datasets_w_fn = {}
        make_fns = [item for item in dir(datasets) if item.startswith("make")]
        load_fns = [item for item in dir(datasets) if item.startswith("load")]
        fetch_fns = [item for item in dir(datasets) if item.startswith("fetch")]
        for item in make_fns:
            datasets_w_fn[item.replace("make_", "")] = item
        for item in load_fns:
            datasets_w_fn[item.replace("load_", "")] = item
        for item in fetch_fns:
            datasets_w_fn[item.replace("fetch_", "")] = item
        return datasets_w_fn

    @classmethod
    def get(cls, name: str):
        datasets_w_fn = cls.list_w_fn()
        if name not in datasets_w_fn:
            raise KeyError(f"Dataset {name} does not exist. Try calling list() for a list of available datasets")
        return cls(datasets_w_fn[name])

    def download(self):
        pass

    def read(self, **read_kwargs):
        return getattr(datasets, self.method)(**read_kwargs)

    @staticmethod
    def clean():
        datasets.clear_data_home()
