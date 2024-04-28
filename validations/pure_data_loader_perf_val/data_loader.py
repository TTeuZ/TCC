from torchvision import transforms, datasets
from datetime import datetime
import torch
import math
import os

class data_loader():
    def __init__(self, config):
        self.transform = transforms.Compose([transforms.Resize(config["img_size"]), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __get_pytorch_dataset(self, subsets):
        dataset_list = []

        for subset in subsets:
            _, path = subset
            dataset_list.append(datasets.ImageFolder(path, self.transform))
        
        return torch.utils.data.ConcatDataset(dataset_list)
    

    def __get_dates(self, ds_path):
        dates = os.listdir(ds_path)
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), f"{ds_path}/{date}") for date in dates]
        
        return sorted(formated_dates, key=lambda x: x[0])
    

    def load_dataset_by_split(self, ds_path="", t_size=1):
        assert os.path.exists(ds_path), "Invalid dataset"

        first_half, second_half = [], []

        subsets = [os.path.join(ds_path, subset) for subset in os.listdir(ds_path)]
        for subset in subsets:
            dates = self.__get_dates(subset)

            divisor = math.ceil(len(dates) * t_size)
            first_half.extend(dates[:divisor])
            second_half.extend(dates[divisor:])
        
        if len(second_half) == 0:
            return self.__get_pytorch_dataset(first_half)
        else:
            return (self.__get_pytorch_dataset(first_half), self.__get_pytorch_dataset(second_half))

    
    def load_dataset_by_subset(self, ds_path=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        datasets = {}
        subsets = os.listdir(ds_path)

        for subset in subsets:
            dates = self.__get_dates(f"{ds_path}/{subset}")
            datasets[subset] = self.__get_pytorch_dataset(dates)

        return datasets