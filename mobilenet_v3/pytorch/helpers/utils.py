import numpy as np
import torch

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.float32)
    w = 128
    h = 128
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.float32).contiguous()

    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.float32)

        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def print_confusion_matrix(cm):
    print("Confusion matrix:")
    
    for i in reversed(range(2)):
        for j in reversed(range(2)):
            print(cm[i][j], end=' ')
        print('\n', end='')