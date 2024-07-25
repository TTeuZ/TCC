import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import create_folder
import numpy as np
import cv2 as cv
import argparse
import json
import os

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def treat_empty_folders(path):
    folders = os.listdir(path)

    for folder in folders:
        empty = len(os.listdir(f"{path}/{folder}/empty"))
        occupied = len(os.listdir(f"{path}/{folder}/occupied"))    

        if empty == 0:
            os.rmdir(f"{path}/{folder}/empty")
            print(f"removing {path}/{folder}/empty")

        if occupied == 0:
            os.rmdir(f"{path}/{folder}/occupied")
            print(f"removing {path}/{folder}/occupied")


def is_complete(obj, keys):
    results = [key in obj for key in keys]
    return not (False in results)


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

# -------------------------------------------------- HELPERS ---------------------------------------------------------

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


def crop_image(cv_image, categories, subset, crop_info, save_path):
    date, time, annotations_rectangle, annotations, offset = crop_info
    bounding_rect = get_bounding_rect(annotations_rectangle)

    for crop in annotations:
        try:
            crop_id = crop["id"]
            category = crop["category_id"]
            image_name = f"{date}_{time}#{subset}#{crop_id}"
            subfolder = f"{date}/{categories[category]['name']}"

            rotated_rect = crop["best_rotated_rect"]

            if not is_inside(rotated_rect, bounding_rect, offset):
                raise Exception("Spot out of boundings or occluded")
            
            cropped = crop_rect(cv_image, rotated_rect)

            print(f"Saving image {image_name} into {save_path}/{subfolder}")
            cv.imwrite(f"{save_path}/{subfolder}/{image_name}.jpg", cropped)
        except Exception as e:
            print(f"Failed to save image {image_name} into {save_path}/{subfolder}: {e}")


def get_images(artifacts, root_path, save_path, subset, offset):
    images, annotations, categories = artifacts
    failed_images = []

    for image in images:
        cv_image = cv.imread(f"{root_path}/{image['file_name']}")

        if cv_image is not None:
            date = '-'.join([str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["date"]])
            time = '_'.join([str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["time"]])
            annotations_rectangle = image["annotationsRectangle"]
            image_annotations = annotations[image["id"]]

            if not os.path.exists(f"{save_path}/{date}"):
                os.mkdir(f"{save_path}/{date}")
                os.mkdir(f"{save_path}/{date}/empty")
                os.mkdir(f"{save_path}/{date}/occupied")

            crop_info = (date, time, annotations_rectangle, image_annotations, offset)
            crop_image(cv_image, categories, subset, crop_info, save_path)
        else:
            failed_images.append(image)

    return failed_images


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


def execute(args, offsets):
    print(f"Generating dataset {args.name} in {args.dest}/{args.name}")

    failed_images = []
    jsons = [file for file in os.listdir(args.root) if ".json" in file]

    for file in jsons:
        subset = file.split("_")[0].upper()
        dest_folder = f"{args.dest}/{args.name}/{subset}"

        print(f"Cropping subset {subset}...")
        create_folder(dest_folder)

        artifacts = get_crop_artifacts(f"{args.root}/{file}")
        failed_images.extend(get_images(artifacts, args.root, dest_folder, subset, offsets[subset]))

        print("Deleting empty folders...")
        treat_empty_folders(dest_folder)

    with open(f"_failed/{args.name}_failed_images.json", "w") as output:
        print("Saving failed images json...")
        json.dump(failed_images, output, indent=2)


def main(args):
    assert os.path.exists(args.root), "invalid source dataset"

    create_folder("_failed")
    create_folder(f"{args.dest}/{args.name}")

    if "PKLot" in args.root:
        offsets = { "PUCPR": 15, "UFPR04": 15, "UFPR05": 15 }
    else:
        offsets = { "CNR-CAMERA-1": 15, "CNR-CAMERA-2": 15, "CNR-CAMERA-3": 15, "CNR-CAMERA-4": 15, "CNR-CAMERA-5": 15, 
                   "CNR-CAMERA-6": 15, "CNR-CAMERA-7": 15, "CNR-CAMERA-8": 15, "CNR-CAMERA-9": 15 }

    execute(args, offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", "-r", type=str, required=True)
    parser.add_argument("--dest", "-d", type=str, required=True)
    parser.add_argument("--name", "-n", type=str, required=True)

    main(parser.parse_args())


# Generate PKLotSegmented dataset: 
# python3 generate_dataset.py -r /home/tteuz/Desktop/datasets/Organized/PKLot -d /media/tteuz/ssd/datasets/PKLot2.0 -n PKLotSegmented

# Generate CNRPark-EXT Segmented dataset: 
# python3 generate_dataset.py -r /home/tteuz/Desktop/datasets/Organized/CNRPark-EXT -d /media/tteuz/ssd/datasets/PKLot2.0 -n CNRParkEXTSegmented