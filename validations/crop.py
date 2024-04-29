import json
import cv2 as cv
import numpy as np

# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/PKLot/PUCPR/Sunny/2012-11-20/2012-11-20_08_54_39.jpg" # -> offset: 9
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/PKLot/UFPR04/Cloudy/2012-12-12/2012-12-12_10_00_05.jpg" # -> offset: 15
IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/PKLot/UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11.jpg" # -> offset: 7

# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera1/2015-11-16_0710.jpg" # -> offset: 
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera2/2015-11-16_0714.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera3/2015-11-16_0715.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera4/2015-11-16_0716.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera5/2015-11-16_0718.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera6/2015-11-16_0717.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera7/2015-11-16_0717.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera8/2015-11-16_0722.jpg" # -> offset: 15
# IMAGE_PATH = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/OVERCAST/2015-11-16/camera9/2015-11-16_0717.jpg" # -> offset: 15

# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/PKLot/pucpr_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/PKLot/ufpr04_spots.json"
JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/PKLot/ufpr05_spots.json"

# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-1_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-2_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-3_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-4_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-5_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-6_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-7_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-8_spots.json"
# JSON_FILE = "/home/tteuz/Desktop/datasets/Organized/CNRPark-EXT/cnr-camera-9_spots.json"

def get_bounding_rect(annotations_rectangle):
    upper_left_x, upper_left_y, width, height = annotations_rectangle

    upper_right_x, upper_right_y = ((upper_left_x + width), upper_left_y)
    lower_left_x, lower_left_y = (upper_left_x, (upper_left_y + height))
    lower_right_x, lower_right_y = (upper_right_x, lower_left_y)

    box_points = np.array([[upper_left_x, upper_left_y],
                       [upper_right_x, upper_right_y],
                       [lower_right_x, lower_right_y],
                       [lower_left_x, lower_left_y]], dtype=np.float32)
    
    return cv.minAreaRect(box_points)


def is_inside(rotated_rect, bounding_rect, offset=0):
    box = cv.boxPoints(rotated_rect)
    bounding_box = cv.boxPoints(bounding_rect)

    for point in box:
        if not (cv.pointPolygonTest(bounding_box, tuple(point), True) + offset) >= 0:
            return False
    return True


def crop_rect(image, rect):
    center, size, angle = rect[0:3]
    height, width = image.shape[0:2]

    if int(angle) >= 45:
        angle -= 90
        size = (size[1], size[0])

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    center, size = tuple(map(int, center)), tuple(map(int, size))

    image_rot = cv.warpAffine(image, rotation_matrix, (width, height))
    image_crop = cv.getRectSubPix(image_rot, size, center)

    return image_crop


def crop_image(cv_image, crop_info):
    annotations_rectangle, annotations = crop_info
    bounding_rect = get_bounding_rect(annotations_rectangle)

    box = np.int64(cv.boxPoints(bounding_rect))
    cv.drawContours(cv_image, [box], 0, (0, 0, 0), 2)

    for crop in annotations:
        # center_x = crop["bbox"][0] + crop["bbox"][2] / 2
        # center_y = crop["bbox"][1] + crop["bbox"][3] / 2
        # rotated_rect = ((center_x, center_y), (crop["bbox"][2], crop["bbox"][3]), 0)
        # print(f"BBOX rotated rect: {rotated_rect}")

        poly = np.array(crop["segmentation"]).reshape((-1, 2)).astype(np.int64)
        rotated_rect = cv.minAreaRect(poly)
        # print(f"POLYrotated rect: {rotated_rect}")

        # if is_inside(rotated_rect, bounding_rect, 9):
        #     cropped = crop_rect(cv_image, rotated_rect)
        #     cv.imshow("batata", cropped)
        #     cv.waitKey(0)

        box = np.int64(cv.boxPoints(rotated_rect))

        if not is_inside(rotated_rect, bounding_rect, 7):
            cv.drawContours(cv_image, [box], 0, (0, 0, 255), 2)
        else :
            cv.drawContours(cv_image, [box], 0, (0, 120, 0), 2)
        
    cv.imshow("batata", cv_image)
    cv.waitKey(0)


def get_crop_artifacts(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    images_ids = set()
    for annotation in data["annotations"]:
        images_ids.add(annotation["image_id"])
    
    annotations = {id: [] for id in images_ids}
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)

    return (data["images"], annotations, data["categories"])


def main():
    artifacts = get_crop_artifacts(JSON_FILE)

    for image in artifacts[0]:
        if image["file_name"] in IMAGE_PATH:
            image_details = image
   
    image_annotations = artifacts[1][image_details["id"]]

    cv_image = cv.imread(IMAGE_PATH)
    annotations_rectangle = image_details["annotationsRectangle"]

    crop_info = (annotations_rectangle, image_annotations)
    crop_image(cv_image, crop_info)

if __name__ == "__main__":
    main()