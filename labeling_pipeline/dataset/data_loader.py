from torchvision import transforms, datasets
from datetime import datetime
import torch
import os

class data_loader():
    def __init__(self, config):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(config["img_size"], antialias=True)])


    def _get_pytorch_dataset(self, date):
        _, _, path = date
        return datasets.ImageFolder(path, self.transform)
    

    def _get_pytorch_full_dataset(self, subsets):
        dataset_list = []

        for subset in subsets:
            dates = self._get_dates(subset)
            dates = [date[2] for date in dates]
            for date in dates:
                dataset_list.append(datasets.ImageFolder(date, self.transform))

        return torch.utils.data.ConcatDataset(dataset_list)


    def _get_dates(self, path):
        dates = os.listdir(path)
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), date, f"{path}/{date}") for date in dates]
        
        return sorted(formated_dates, key=lambda x: x[0])

    
    def get_subset_from_dataset(self, ds_path="", subset=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        path = f"{ds_path}/{subset}"
        dates = self._get_dates(path)

        datasets = {}
        for date in dates:
            datasets[date[1]] = self._get_pytorch_dataset(date)

        return datasets
    

    def load_dataset_with_val_subset(self, ds_path="", subset=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        val_subset = [os.path.join(ds_path, subset)]
        subsets = [os.path.join(ds_path, subset) for subset in sorted(os.listdir(ds_path))]
        train_subsets = [subset for subset in subsets if val_subset[0] not in subset]

        return self._get_pytorch_full_dataset(train_subsets), self._get_pytorch_full_dataset(val_subset)