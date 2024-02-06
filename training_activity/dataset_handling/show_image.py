import cv2 as cv

if __name__ == "__main__":
    image = cv.imread("../PKLot/PKLot/UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11.jpg")
    cv.imshow("Display window", image)
    cv.waitKey(0)
    cv.destroyAllWindows()