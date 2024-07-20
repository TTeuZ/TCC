import numpy as np
import json
import os
import re

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def get_cm(cm):
    temp = re.findall(r'\d+', cm)
    temp = np.array(temp, dtype=int).reshape(-1, 2)

    return temp


def get_related_models_json(path, dataset):
    jsons = os.listdir(f"{path}/jsons")
    jsons = [json.load(open(f"{path}/jsons/{file}", "r")) for file in jsons]
    jsons = [file for file in jsons if dataset in file["dataset"]["train"]]

    return jsons