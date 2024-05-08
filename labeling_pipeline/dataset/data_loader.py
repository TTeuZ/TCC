from torchvision import transforms, datasets
from datetime import datetime
import numpy as np
import torch
import os

class data_loader():
    def __init__(self, config):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(config["img_size"], antialias=True)])


    def __get_pytorch_dataset(self, date):
        _, _, path = date
        return datasets.ImageFolder(path, self.transform)


    def __get_dates(self, path):
        dates = os.listdir(path)
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), date, f"{path}/{date}") for date in dates]
        
        return sorted(formated_dates, key=lambda x: x[0])

    
    def get_subset_from_dataset(self, ds_path="", subset=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        path = f"{ds_path}/{subset}"
        dates = self.__get_dates(path)

        datasets = {}
        for date in dates:
            datasets[date[1]] = self.__get_pytorch_dataset(date)

        return datasets