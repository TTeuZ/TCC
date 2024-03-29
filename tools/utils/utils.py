import numpy as np
import os
import re

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def get_cm(cm):
    temp = re.findall(r'\d+', cm)
    temp = np.array(temp, dtype=int).reshape(-1, 2)

    return temp