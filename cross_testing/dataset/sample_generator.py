from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import transforms
from collections import Counter
from datetime import datetime
from PIL import Image
import random

FLATTEN_LINIAR = 5

class sample_dataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample, label = self.samples[index]
        image = Image.open(sample)

        return self.transform(image), label


class sample_generator():
    def __init__(self, dl_config):
        self.config = dl_config
        self.date_formats = ["%Y-%m-%d_%H_%M", "%Y-%m-%d_%H_%M_%S"]

        self.subsets_spots = defaultdict(int)
        self.source = defaultdict(list)
    

    def _parse_date(self, date):
        for fmt in self.date_formats:
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue


    def _get_random(self, source, qty):
        if len(source) > qty:
            return random.sample(source, qty)
        else:
            return source
        
    
    def _flatten_sample(self, data):
        labels = [item[1] for item in data]

        count = Counter(labels)
        diff = abs(count[0] - count[1])
        diff_perc = round((diff / (count[0] + count[1])) * 100)

        if diff_perc > FLATTEN_LINIAR:
            remove_class = 0 if count[0] > count[1] else 1
            indexes = [index for index, label in enumerate(labels) if label == remove_class]
            indexes_to_remove = random.sample(indexes, diff)

            temp = [item for index, item in enumerate(data) if index not in indexes_to_remove]
        
        return temp


    def build(self, src_dataset):
        days_by_subset = defaultdict(list)
        for dataset in src_dataset.datasets:
            subset = dataset.root.split("/")[-2]
            days_by_subset[subset].append(dataset)

        for subset, days in days_by_subset.items():
            hours_qty = 0

            for day in days:
                samples_by_hour = defaultdict(list)
                for sample in day.samples:
                    hour = sample[0].split("/")[-1].split("#")[0]
                    samples_by_hour[hour].append(sample)

                morning = {0: [], 1: []}
                afternoon = {0: [], 1: []}
                for hour, sample_list in samples_by_hour.items():
                    self.subsets_spots[subset] += len(sample_list)
                    hours_qty += 1

                    date = self._parse_date(hour)
                    for sample in sample_list:
                        if date.hour < 12:
                            morning[sample[1]].append(sample)
                        else:
                            afternoon[sample[1]].append(sample)
                
                self.source[subset].append({"morning": morning, "afternoon": afternoon})
            self.subsets_spots[subset] = round(self.subsets_spots[subset] / hours_qty)

    
    def get_sample(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.config["img_size"], antialias=True)])
        data = []

        for subset, days in self.source.items():
            spots_qty = self.subsets_spots[subset]

            for day in days:
                data.extend(self._get_random(day["morning"][0], spots_qty))
                data.extend(self._get_random(day["morning"][1], spots_qty))
                data.extend(self._get_random(day["afternoon"][0], spots_qty))
                data.extend(self._get_random(day["afternoon"][1], spots_qty))

        data = self._flatten_sample(data)
        return sample_dataset(data, transform)