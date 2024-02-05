import cv2 as cv
import numpy as np
import xml.etree.ElementTree as et

#TODO Using the XML file, cropp the image in each parking space and save it in one directory
#TODO Validate the cropped image. Use the images available in the dataset

def crop_rect(image, rect):
    center, size, angle = rect[0:3]
    height, width = image.shape[0:2]

    print(center, size, angle)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    center, size = tuple(map(int, center)), tuple(map(int, size))

    image_rot = cv.warpAffine(image, rotation_matrix, (width, height))
    image_crop = cv.getRectSubPix(image_rot, size, center)

    return image_crop


if __name__ == "__main__":
    image = cv.imread("../PKLot/PKLot/UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11.jpg")

    tree = et.parse("../PKLot/PKLot/UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11.xml")
    root = tree.getroot()

    # cnt = np.array([[[608, 613]], [[741, 654]], [[775, 582]], [[608, 526]]])
    # rotaded_rect = cv.minAreaRect(cnt)
    rotaded_rect = (tuple(map(int, root[0][0][0].attrib.values())), tuple(map(int, root[0][0][1].attrib.values())), int(root[0][0][2].attrib["d"]))

    cropped = crop_rect(image, rotaded_rect)

    cv.imshow("Display window", cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()


    # cnt = np.array([[[945, 223]], [[1013, 236]], [[1005, 307]], [[935, 294]]])
    # rotaded_rect = cv.minAreaRect(cnt)
    rotaded_rect = (tuple(map(int, root[22][0][0].attrib.values())), tuple(map(int, root[22][0][1].attrib.values())), int(root[22][0][2].attrib["d"]))

    cropped = crop_rect(image, rotaded_rect)

    cv.imshow("Display window", cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()



    # for child in root:
    #     rotaded_rect = (tuple(map(int, child[0][0].attrib.values())), tuple(map(int, child[0][1].attrib.values())), int(child[0][2].attrib["d"]))
    #     cropped = crop_rect(image, rotaded_rect)

    #     cv.imshow("Display window", cropped)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()