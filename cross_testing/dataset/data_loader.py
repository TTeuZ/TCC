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


    def _load_image(self, images):
        processed_images = []

        for image in images:
            processed_images.append(cv.imread(image))
        
        return processed_images


    def _process_directory(self, args):
        _, path = args

        images, targets = [], []
        for class_name in os.listdir(path):
            full_path = f"{path}/{class_name}"

            for image in os.listdir(full_path):
                images.append(f"{full_path}/{image}")
                targets.append(CLASSES[class_name])
        
        return (images, targets)


    def _get_dataset(self, paths):
        images, targets = [], []

        pool = Pool()
        results = pool.map(self._process_directory, paths)
        pool.close(), pool.join()

        images_by_day = [item[0] for item in results]
        targets_by_day = [item[1] for item in results]
        
        pool = Pool()
        processed_images = pool.map(self._load_image, [item for item in images_by_day])
        pool.close(), pool.join()

        for processed in processed_images:
            images.extend(processed)

        for target in targets_by_day:
            targets.extend(target)
        targets = np.array(targets)

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