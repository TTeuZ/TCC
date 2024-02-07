from skimage.feature import local_binary_pattern
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
import cv2 as cv

def get_lbp_histogram(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray_image, 8, 1)
    (hist, _) = np.histogram(lbp.ravel(), bins=256, range=(0, 255))

    return hist


if __name__ == "__main__":
    images = [f for f in listdir("../images") if isfile(join("../images", f))]

    metadatas = []
    features = []

    for image in images:
        temp = cv.imread(f"../images/{image}")

        metadata = image.split("#")
        status = metadata.pop(2)
        metadata[len(metadata) - 1] = metadata[len(metadata) - 1].split(".")[0]
        metadatas.append(metadata)

        hist = get_lbp_histogram(temp)
        hist = np.insert(hist, len(hist), 1 if status == "Occupied" else 0)
        features.append(hist)

    pd_metadata = pd.DataFrame(columns=["dataset", "weather", "date", "hour", "parking_space"], data=metadatas)
    pd_features = pd.DataFrame(features)

    final_features = pd.concat([pd_metadata, pd_features], axis=1)
    final_features.to_csv("../features/features.csv", index=False)