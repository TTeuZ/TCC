import numpy as np
import torch

def fast_collate(batch, config):
    images = [image[0] for image in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.long)

    width, height = config["img_size"]
    tensor = torch.zeros((len(images), 3, height, width), dtype=torch.float32).contiguous()

    for index, image in enumerate(images):
        nump_array = np.asarray(image, dtype=np.float32)

        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2)
        tensor[index] += torch.from_numpy(nump_array)

    return tensor, targets


class data_prefetcher():
    def __init__(self, loader, normalize_data):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.normalize_date = normalize_data

        if self.normalize_date:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        else:
            self.mean = torch.tensor([0.0, 0.0, 0.0]).cuda().view(1,3,1,1)
            self.std = torch.tensor([1.0, 1.0, 1.0]).cuda().view(1,3,1,1)

        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

            self.next_input = self.next_input.sub_(self.mean).div_(self.std)


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target

        if input is not None:
            input.record_stream(torch.cuda.current_stream())

        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        
        self.preload()
        return input, target