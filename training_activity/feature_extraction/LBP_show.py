from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt 
import numpy as np
import cv2 as cv


def get_lbp_and_histogram(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray_image, 8, 1)
    (hist, _) = np.histogram(lbp.ravel(), bins=256, range=(0, 255))

    return (lbp, hist)


if __name__ == "__main__":
    image = cv.imread("../images/PUCPR#Cloudy#Empty#2012-09-12#06_05_16#1")
    lbp, hist = get_lbp_and_histogram(image)

    print(hist)

    plt.style.use("ggplot")
    (fig, ax) = plt.subplots()
    fig.suptitle("Local Binary Patterns")

    ax.hist(lbp.ravel(), density=True, bins=20, range=(0, 255))
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 0.03])

    plt.show()

    cv.imshow("LBP", lbp.astype("uint8"))
    cv.waitKey(0)