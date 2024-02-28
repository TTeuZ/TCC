import argparse
import json
import os

def crop_images(artifacts, save_path):
    print()


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


def main(args):
    print(f"Generating dataset {args.name} in {args.dest}/{args.name}")

    jsons = [file for file in os.listdir(args.root) if ".json" in file]
    for json in jsons:
        print(f"Cropping subset {json}...")

        artifacts = get_crop_artifacts(f"{args.root}/{json}")
        crop_images(artifacts, f"{args.dest}/{args.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", "-r", type=str)
    parser.add_argument("--dest", "-d", type=str)
    parser.add_argument("--name", "-n", type=str)

    main(parser.parse_args())

# Generate PKLotSegmented dataset: 
# python3 dataset/generate_dataset.py -r /home/tteuz/Desktop/TCC/datasets/PKLot2.0/PKLot -d /home/tteuz/Desktop/TCC/datasets/PKLot2.0 -n PKLotSegmented

# Generate CNRPark-EXT Segmented dataset: 
# python3 dataset/generate_dataset.py -r /home/tteuz/Desktop/TCC/datasets/PKLot2.0/CNRPark-EXT -d /home/tteuz/Desktop/TCC/datasets/PKLot2.0 -n CNRPartEXTSegmented