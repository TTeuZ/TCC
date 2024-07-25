from torch.utils.data import Dataset
import numpy as np

class raspberry_dataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform
        self.labels = np.zeros(len(images))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]