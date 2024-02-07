import cv2 as cv
import numpy as np
import xml.etree.ElementTree as et

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
    image = cv.imread("../../PKLot/PKLot/UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11.jpg")
    tree = et.parse("../../PKLot/PKLot/UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11.xml")
    root = tree.getroot()

    for child in root:
        # contors = []
        # for contor in child[1]:
        #     contors.append([[int(contor.attrib["x"]), int(contor.attrib["y"])]])
        
        # contors.append([[int(child[1][0].attrib["x"]), int(child[1][0].attrib["y"])]])
        # contors.append([[int(child[1][1].attrib["x"]), int(child[1][1].attrib["y"])]])
            
        # contors = np.array(contors)
        # rotaded_rect = cv.minAreaRect(contors)
        # print(rotaded_rect)

        rotaded_rect = (tuple(map(int, child[0][0].attrib.values())), tuple(map(int, child[0][1].attrib.values())), int(child[0][2].attrib["d"]))

        box = cv.boxPoints(rotaded_rect)
        box = np.int64(box)
        cv.drawContours(image, [box], 0, (0, 0, 255), 2)


    # cnt = np.array([[[608, 613]], [[741, 654]], [[775, 582]], [[608, 526]]])
    # rotaded_rect = cv.minAreaRect(cnt)
    # print(rotaded_rect)


    cv.imshow("Display window", image)
    cv.waitKey(0)
