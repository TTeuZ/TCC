from collections import Counter
import numpy as np
import random

FLATTEN_LINIAR = 5

class data_leveler():
    def __remove_images_from_dataset(self, dataset, indexes_to_remove):
        for index in sorted(indexes_to_remove, reverse=True):
            del dataset.data[index]
            dataset.targets = np.delete(dataset.targets, index)


    def __flatten_dataset(self, dataset, flatter_object):
        indexes = [index for index, target in enumerate(dataset.targets) if target == flatter_object[1]]
        indexes_to_remove = random.sample(indexes, flatter_object[0])

        self.__remove_images_from_dataset(dataset, indexes_to_remove)


    def __get_flatter_object(self, dataset):
        class_0, class_1 = 0, 0

        class_count = Counter(dataset.targets)
        class_0 += class_count[0]
        class_1 += class_count[1]
        
        diff = abs(class_0 - class_1)
        diff_perc = round((diff / (class_0 + class_1)) * 100)

        return (diff, 0, diff_perc) if class_0 > class_1 else (diff, 1, diff_perc)


    def flatten_dataset(self, dataset):
        flatter_object = self.__get_flatter_object(dataset)

        if flatter_object[2] > FLATTEN_LINIAR:
            self.__flatten_dataset(dataset, flatter_object)