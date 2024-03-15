from matplotlib import pyplot as plt 
import numpy as np
import cv2 as cv

class lbp_extractor:
    def __init__(self):
        self.multiplier = np.array([1, 2, 4, 8, 16, 32, 64, 128])


    def __get_pixel(self, image, x, y):
        sub_matrix = image[np.ix_([x-1, x, x+1],[y-1, y, y+1])]
        pixels = sub_matrix.ravel()

        center = pixels[4]
        pixels = np.delete(pixels, 4)
        pixels = np.where(pixels >= center, 1, 0)

        return np.dot(self.multiplier.T, pixels)


    def __local_binary_pattern(self, image):
        height, width = image.shape
        image = np.pad(image, [(1, 1), (1, 1)], mode="constant")

        lbp_image = np.zeros((height, width), np.uint8)
        for i in range(1, (height + 1)): 
            for j in range(1, (width + 1)): 
                lbp_image[i - 1][j - 1] = self.__get_pixel(image, i, j)
        
        return lbp_image


    def get_lbp_image(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        return self.__local_binary_pattern(gray_image)


    def get_lbp_histogram(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        lbp_image = self.__local_binary_pattern(gray_image)
        (hist, _) = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256), range=(0, 255))

        return hist


if __name__ == "__main__":
    extractor = lbp_extractor()
    image = cv.imread("../images/PUCPR#Rainy#Occupied#2012-10-11#07_58_36#69.jpg")

    lbp = extractor.get_lbp_image(image)
    hist = extractor.get_lbp_histogram(image)

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

