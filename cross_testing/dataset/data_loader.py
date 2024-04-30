import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torchvision import transforms, datasets
from tools.dataset.memory_dataset import memory_dataset
from multiprocessing import Pool
from datetime import datetime
import numpy as np
import cv2 as cv
import math
import os

CLASSES = {"empty": 0, "occupied": 1}

class data_loader():
    def __init__(self, config):
        self.transform = transforms.Compose([transforms.Resize(config["img_size"]), transforms.ToTensor()])


    def _load_image(self, image_path):
        return cv.imread(image_path)


    def _process_directory(self, args):
        path, class_name = args

        images = []
        for image in os.listdir(path):
            image_path = f"{path}/{image}"
            images.append((image_path, class_name))
        
        return images


    def _get_dataset(self, paths):
        all_paths, images, targets = [], [], []

        for _, path in paths:
            for class_name in os.listdir(path):
                all_paths.append((f"{path}/{class_name}", class_name))

        pool = Pool()
        results = pool.map(self._process_directory, all_paths)
        pool.close(), pool.join()

        image_paths = [item for sublist in results for item in sublist]

        pool = Pool()
        images = pool.map(self._load_image, [item[0] for item in image_paths])
        pool.close(), pool.join()

        targets = [CLASSES[item[1]] for item in image_paths]

        targets =  np.array(targets)
        return memory_dataset(images, targets, self.transform)


    def _get_dates(self, ds_path):
        dates = os.listdir(ds_path)
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), f"{ds_path}/{date}") for date in dates]
        
        return sorted(formated_dates, key=lambda x: x[0])


    def load_dataset_by_split(self, ds_path="", t_size=1):
        assert os.path.exists(ds_path), "Invalid dataset"

        subsets = [os.path.join(ds_path, subset) for subset in sorted(os.listdir(ds_path))]
        
        first_half, second_half = [], []
        for subset in subsets:
            dates = self._get_dates(subset)

            divisor = math.ceil(len(dates) * t_size)
            first_half.extend(dates[:divisor])
            second_half.extend(dates[divisor:])
    
        if len(second_half) == 0:
            return self._get_dataset(first_half)
        else:
            return (self._get_dataset(first_half), self._get_dataset(second_half))
        

    def load_dataset_by_subset(self, ds_path=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        datasets = {}
        subsets = os.listdir(ds_path)

        for subset in subsets:
            dates = self._get_dates(f"{ds_path}/{subset}")
            datasets[subset] = self._get_dataset(dates)

        return datasets