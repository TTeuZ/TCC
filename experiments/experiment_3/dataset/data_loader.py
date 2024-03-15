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

        dates = []
        subsets = os.listdir(ds_path)

        for subset in subsets:
            dates.extend(self.__get_dates(f"{ds_path}/{subset}"))
        dates.sort(key=lambda x: x[0])

        divisor = math.ceil(len(dates) * t_size)
        first_half = dates[:divisor]
        second_half = dates[divisor:]

        if len(second_half) == 0:
            return self.__get_pytorch_dataset(first_half)
        else:
            return (self.__get_pytorch_dataset(first_half), self.__get_pytorch_dataset(second_half))

    
    def load_dataset_as_test(self, ds_path=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        datasets = {}
        subsets = os.listdir(ds_path)

        for subset in subsets:
            dates = self.__get_dates(f"{ds_path}/{subset}")
            datasets[subset] = self.__get_pytorch_dataset(dates)

        return datasets