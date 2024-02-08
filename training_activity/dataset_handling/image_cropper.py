import xml.etree.ElementTree as et
from os import listdir
import numpy as np
import cv2 as cv

def crop_rect(image, rect):
    center, size, angle = rect[0:3]
    height, width = image.shape[0:2]

    if int(angle) >= 45:
        angle = angle - 90
        temp = size
        size = (temp[1], temp[0])

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    center, size = tuple(map(int, center)), tuple(map(int, size))

    image_rot = cv.warpAffine(image, rotation_matrix, (width, height))
    image_crop = cv.getRectSubPix(image_rot, size, center)

    return image_crop


if __name__ == "__main__":
    root_dir = "../../PKLot/PKLot"

    subsets = listdir(root_dir)
    for subset in subsets:
        climatic_condition = listdir(f"{root_dir}/{subset}")
        for weather in climatic_condition:
            dates = listdir(f"{root_dir}/{subset}/{weather}")
            for date in dates:
                xmls = [file for file in listdir(f"{root_dir}/{subset}/{weather}/{date}") if ".xml" in file]
                for xml in xmls:
                    jpg = f"{root_dir}/{subset}/{weather}/{date}/{xml.split(".xml")[0]}.jpg"
                    full_xml = f"{root_dir}/{subset}/{weather}/{date}/{xml}"
                    hour = xml.split(".")[0].split("-")[2][3:]

                    image =  cv.imread(jpg)
                    tree = et.parse(full_xml)
                    root = tree.getroot()

                    for child in root:
                        if "occupied" in child.attrib.keys():
                            contors = []
                            for contor in child[1]:
                                contors.append([[int(contor.attrib["x"]), int(contor.attrib["y"])]])

                            contors = np.array(contors)
                            rotaded_rect = cv.minAreaRect(contors)

                            # rotaded_rect = (tuple(map(int, child[0][0].attrib.values())), tuple(map(int, child[0][1].attrib.values())), int(child[0][2].attrib["d"]))
                            cropped_lot_space = crop_rect(image, rotaded_rect)

                            image_name = f"{subset}#{weather}#{"Occupied" if int(child.attrib["occupied"]) == 1 else "Empty"}#{date}#{hour}#{child.attrib["id"]}.jpg"
                            print(f"Saving image: {image_name}")

                            cv.imwrite(f"../images/{image_name}", cropped_lot_space)
