from collections import Counter
import numpy as np
import random
import torch
import copy

FLATTEN_LINIAR = 5

# Works only with Pytorch ConcatDatasets and new labels array
class data_leveler():
    def __remove_images_from_dataset(self, dataset, labels, indexes_to_remove):
        current_index = 0

        temp_dataset = []
        for _, subset in enumerate(dataset.datasets):
            subset_len = len(subset)
            indexes_in_subset = [index - current_index for index in indexes_to_remove if current_index <= index < current_index + subset_len]

            temp_subset = copy.deepcopy(subset)
            for index in sorted(indexes_in_subset, reverse=True):
                del temp_subset.imgs[index]
                del temp_subset.targets[index]
            temp_subset.samples = temp_subset.imgs

            temp_dataset.append(temp_subset)
            current_index += subset_len

        mask = np.ones(len(labels), dtype=bool)
        mask[indexes_to_remove] = False
         
        return (torch.utils.data.ConcatDataset(temp_dataset), labels[mask])


    def __flatten_dataset(self, dataset, labels, flatter_object):
        indexes = [index for index, label in enumerate(labels) if label == flatter_object[1]]
        indexes_to_remove = random.sample(indexes, flatter_object[0])

        return self.__remove_images_from_dataset(dataset, labels, indexes_to_remove)


    def __get_flatter_object(self, labels):
        class_0, class_1 = 0, 0

        class_count = Counter(labels)
        class_0 += class_count[0]
        class_1 += class_count[1]
        
        diff = abs(class_0 - class_1)
        diff_perc = round((diff / (class_0 + class_1)) * 100)

        return (diff, 0, diff_perc) if class_0 > class_1 else (diff, 1, diff_perc)


    def flatten_dataset(self, dataset, labels):
            flatter_object = self.__get_flatter_object(labels)

            return (dataset, labels) if flatter_object[2] < FLATTEN_LINIAR else self.__flatten_dataset(dataset, labels, flatter_object)