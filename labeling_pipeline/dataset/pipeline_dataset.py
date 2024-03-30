from torch.utils.data import Dataset

class pipeline_dataset(Dataset):
    def __init__(self, original_dataset, new_labels):
        self.original_dataset = original_dataset
        self.labels = new_labels
    

    def __len__(self):
        return len(self.original_dataset)


    def __getitem__(self, index):
            original_sample = self.original_dataset[index]
            modified_label = self.labels[index]
            
            return (original_sample[0], modified_label)