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


    def __get_subsets(self, ds_path, dates):
        return [(datetime.strptime(date, '%Y-%m-%d').date(), f"{ds_path}/{date}") for date in dates]


    def __get_full_dataset(self, ds_path):
        dates = os.listdir(ds_path)
        subsets = self.__get_subsets(ds_path, dates)

        return self.__get_pytorch_dataset(subsets)

    
    def __get_split_dataset(self, ds_path, t_size):
        dates = os.listdir(ds_path)

        subsets = self.__get_subsets(ds_path, dates)
        subsets.sort(key=lambda x: x[0])

        divisor = math.ceil(len(subsets) * t_size)
        first_half = subsets[:divisor]
        second_half = subsets[divisor:]

        return (self.__get_pytorch_dataset(first_half), self.__get_pytorch_dataset(second_half))


    def load_dataset(self, ds_path="", t_size=1):
        assert(os.path.exists(ds_path))

        if t_size == 1:
            return self.__get_full_dataset(ds_path)
        else:
            return self.__get_split_dataset(ds_path, t_size)