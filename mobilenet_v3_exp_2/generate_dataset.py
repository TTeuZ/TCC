import numpy as np
import cv2 as cv
import argparse
import json
import os

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


def crop_image(cv_image, annotations, categories, date, time, subset, save_path):
    for crop in annotations:
        try:
            crop_id = crop["id"]
            category = crop["category_id"]

            poly = np.array(crop["segmentation"])
            poly = np.reshape(poly, (-1, 2))
            poly = np.int64(poly)

            rotaded_rect = cv.minAreaRect(poly)

            box = cv.boxPoints(rotaded_rect)
            box = np.int64(box)

            cropped = crop_rect(cv_image, rotaded_rect)

            image_name = f"{date}_{time}#{subset}#{crop_id}"
            subfolder = f"{date}/{categories[category]['name']}"
            print(f"Saving image {image_name} into {save_path}/{subfolder}")

            cv.imwrite(f"{save_path}/{subfolder}/{image_name}.jpg", cropped)
        except:
            print(f"Failed save image {image_name} into {save_path}/{subfolder}")


def get_images(artifacts, root_path, save_path):
    images, annotations, categories = artifacts
    failed_images = []

    for image in images:
        cv_image = cv.imread(f"{root_path}/{image['file_name']}")

        if cv_image is not None and is_complete(image, ["id", "date", "time", "subset"]):
            date = [str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["date"]]
            time = [str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["time"]]

            date = '-'.join(date)
            time = '_'.join(time)
            subset = image["subset"]
            image_annotations = annotations[image["id"]]

            if not os.path.exists(f"{save_path}/{date}"):
                os.mkdir(f"{save_path}/{date}")
                os.mkdir(f"{save_path}/{date}/empty")
                os.mkdir(f"{save_path}/{date}/occupied")

            crop_image(cv_image, image_annotations, categories, date, time, subset, save_path)
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


def execute(args):
    print(f"Generating dataset {args.name} in {args.dest}/{args.name}")

    failed_images = []

    jsons = [file for file in os.listdir(args.root) if ".json" in file]
    for file in jsons:
        print(f"Cropping subset {file}...")

        artifacts = get_crop_artifacts(f"{args.root}/{file}")
        failed_images.extend(get_images(artifacts, args.root, f"{args.dest}/{args.name}"))
    
    with open(f"_failed/{args.name}_failed_images.json", "x") as output:
        json.dump(failed_images, output, indent=2)

    treat_empty_folders(f"{args.dest}/{args.name}")


def main(args):
    assert os.path.exists(args.root), "invalid source dataset"

    if not os.path.exists("_failed"):
        os.mkdir("_failed")

    if not os.path.exists(f"{args.dest}/{args.name}"):
        os.mkdir(f"{args.dest}/{args.name}")

    execute(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", "-r", type=str, required=True)
    parser.add_argument("--dest", "-d", type=str, required=True)
    parser.add_argument("--name", "-n", type=str, required=True)

    main(parser.parse_args())


# Generate PKLotSegmented dataset: 
# python3 generate_dataset.py -r /home/tteuz/Desktop/TCC/datasets/PKLot2.0/PKLot -d /home/tteuz/Desktop/TCC/datasets/PKLot2.0 -n PKLotSegmented

# Generate CNRPark-EXT Segmented dataset: 
# python3 generate_dataset.py -r /home/tteuz/Desktop/TCC/datasets/PKLot2.0/CNRPark-EXT -d /home/tteuz/Desktop/TCC/datasets/PKLot2.0 -n CNRPartEXTSegmented