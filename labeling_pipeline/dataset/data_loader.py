from dataset.pipeline_dataset import pipeline_dataset
from torchvision import transforms, datasets
from datetime import datetime
import numpy as np
import torch
import os

class data_loader():
    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize((128, 128))])


    def __get_pytorch_dataset(self, date):
        _, _, path = date
        return datasets.ImageFolder(path, self.transform)


    def __format_dates(self, path, dates):
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), date, f"{path}/{date}") for date in dates]
        formated_dates.sort(key=lambda x: x[0])
        
        return formated_dates

    
    def get_subset_from_dataset(self, ds_path="", subset=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        path = f"{ds_path}/{subset}"
        dates = os.listdir(path)
        dates = self.__format_dates(path, dates)

        datasets = {}
        for date in dates:
            datasets[date[1]] = self.__get_pytorch_dataset(date)

        return datasets


    def concat_dataset_with_new_labels(self, datasets, new_labels):
        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        concat_labels = np.concatenate(new_labels)

        return pipeline_dataset(concat_dataset, concat_labels)
    

    def concat_dataset(self, datasets):
        return torch.utils.data.ConcatDataset(datasets)