from torch.utils.data import Dataset

class memory_dataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index], self.targets[index]