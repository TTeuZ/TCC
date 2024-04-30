from torch.utils.data import Dataset

class memory_dataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return self.transform(img), target