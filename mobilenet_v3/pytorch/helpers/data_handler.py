import os
import math
from datetime import datetime

import torch
from torchvision import transforms, datasets

def generate_pytorch_dataset(dates, transform):
    dataset_list = []
    for date in dates:
        _, path = date
        dataset_list.append(datasets.ImageFolder(path, transform))

    final_dataset = torch.utils.data.ConcatDataset(dataset_list)
    return final_dataset


def get_train_val_datasets(transform):
    root_dir = "../../PKLot/PKLotSegmented"
    subsets = ["UFPR04", "UFPR05"]

    all_dates = []
    for subset in subsets:
        climatic_condition = os.listdir(f"{root_dir}/{subset}")
        for weather in climatic_condition:
            dates = os.listdir(f"{root_dir}/{subset}/{weather}")
            temp = [(datetime.strptime(date, '%Y-%m-%d').date(), f"{root_dir}/{subset}/{weather}/{date}") for date in dates]

            all_dates.extend(temp)

    all_dates.sort(key=lambda x: x[0])

    train_days = math.ceil(len(all_dates) * 0.7)
    train = all_dates[:train_days]
    validation = all_dates[train_days:]

    final_train = generate_pytorch_dataset(train, transform)
    final_validation = generate_pytorch_dataset(validation, transform)

    return final_train, final_validation


def get_test_dataset(transform):
    root_dir = "../../PKLot/PKLotSegmented/PUC"

    test_ds = []
    climatic_condition = os.listdir(f"{root_dir}")
    for weather in climatic_condition:
        dates = os.listdir(f"{root_dir}/{weather}")
        for date in dates:
            path = f"{root_dir}/{weather}/{date}"
            test_ds.append(datasets.ImageFolder(path, transform))

    final_test = torch.utils.data.ConcatDataset(test_ds)
    return final_test


def get_datasets():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.ToTensor(), # To slow
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # To slow
    ])

    final_train, final_validation = get_train_val_datasets(transform)
    final_test = get_test_dataset(transform)

    return (final_train, final_validation, final_test)
    