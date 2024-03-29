from torchvision import transforms, datasets
from datetime import datetime
import torch
import math
import os

class data_loader():
    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((128, 128))])


    def __get_pytorch_dataset(self, subsets):
        dataset_list = []

        for subset in subsets:
            _, path = subset
            dataset_list.append(datasets.ImageFolder(path, self.transform))
        
        return torch.utils.data.ConcatDataset(dataset_list)
    

    def __get_dates(self, ds_path):
        dates = os.listdir(ds_path)
        return [(datetime.strptime(date, '%Y-%m-%d').date(), f"{ds_path}/{date}") for date in dates]
    

    def load_dataset_as_train(self, ds_path="", t_size=1):
        assert os.path.exists(ds_path), "Invalid dataset"

        all_dates = []
        subsets = [os.path.join(ds_path, subset) for subset in os.listdir(ds_path)]
        for subset in subsets:
            all_dates.extend(self.__get_dates(subset))

        dates_set = sorted(set(all_dates), key=lambda x: x[0])
        final_dates = {date[0]: [] for date in dates_set}
        for date in all_dates:
            final_dates[date[0]].append(date)

        train_ds, val_ds = [], []
        divisor = math.ceil(len(final_dates) * t_size)
        for index, date in enumerate(final_dates.values()):
            if index < divisor:
                train_ds.extend(date)
            else:
                val_ds.extend(date)
        
        if len(val_ds) == 0:
            return self.__get_pytorch_dataset(train_ds)
        else:
            return (self.__get_pytorch_dataset(train_ds), self.__get_pytorch_dataset(val_ds))

    
    def load_dataset_as_test(self, ds_path=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        datasets = {}
        subsets = os.listdir(ds_path)

        for subset in subsets:
            dates = self.__get_dates(f"{ds_path}/{subset}")
            datasets[subset] = self.__get_pytorch_dataset(dates)

        return datasets