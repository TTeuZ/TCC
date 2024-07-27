import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.raspberry_dataset import raspberry_dataset
from torchvision import transforms
import numpy as np
import cv2 as cv
import importlib
import argparse
import torch
import json
import time

OFFSET = 15

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def get_models(model_module, base, model_config):
    loaded_models = dict()
    models = sorted(os.listdir(f"{base}/models"))
    jsons = [json.load(open(f"{base}/jsons/{file}", "r")) for file in sorted(os.listdir(f"{base}/jsons"))]

    for index, model in enumerate(models):
        model_name, model_path = model.split(".")[0].split("_")[1], f"{base}/models/{model}"
        
        temp_model = model_module.model(pre_trained=False, config=model_config)
        temp_model.load_state_dict(torch.load(model_path, map_location=model_config["device"]))

        loaded_models[model_name] = {"model": temp_model, "threshold": jsons[index]["model"]["metrics"]["threshold"]}

    return loaded_models 


def get_datasets(config):
    datasets = dict()

    root_path = config["dataset"]["path"]
    jsons = sorted([file for file in os.listdir(root_path) if file.endswith(".json")])
    for file in jsons:
        subset = file.split("_")[0].upper()
        images, annotations = get_crop_artifacts(f"{root_path}/{file}")

        dates = sorted(set(image["file_name"].split("/")[1] for image in images))
        days_to_exclude = dates[:config["dataset"]["exclude_days"]]

        final_images = [image for image in images if image["file_name"].split("/")[1] not in days_to_exclude]
        datasets[subset] = {"images": final_images, "annotations": annotations}

    return datasets


def get_crop_artifacts(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    images_ids = set()
    for annotation in data["annotations"]:
        images_ids.add(annotation["image_id"])
    
    annotations = {id: [] for id in images_ids}
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)

    return (sorted(data["images"], key=lambda x: x["file_name"].split("/")[1]), annotations)


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


def crop_image(cv_image, annotations_rectangle, annotations):
    bounding_rect = get_bounding_rect(annotations_rectangle)
    crops = []

    for crop in annotations:
        rotated_rect = crop["best_rotated_rect"]
        
        if is_inside(rotated_rect, bounding_rect, OFFSET):
            crops.append(crop_rect(cv_image, rotated_rect))
    
    return crops


def classify_batch(model, dl_config, transform, spots):
    dataset = raspberry_dataset(spots, transform)
    batch_loader = torch.utils.data.DataLoader(dataset, batch_size=len(spots), shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)
    
    _, _ = model["model"].predict(batch_loader, model["threshold"])


def execute_for_subset(config, dl_config, transform, model, subset):
    total_times, crop_images_times, classify_images_times = [], [], []
    total_per_spot, crop_images_per_spot, classify_images_per_spot = [], [], []
    root_path = config["dataset"]["path"]
    total_spots = 0

    for image in subset["images"]:
        start_total_time = time.time()
        annotations = subset["annotations"][image["id"]]

        start_crop_time = time.time()
        cv_image = cv.imread(f"{root_path}/{image['file_name']}")
        spots = crop_image(cv_image, image["annotationsRectangle"], annotations)
        end_crop_time = time.time()

        start_classify_time = time.time()
        classify_batch(model, dl_config, transform, spots)
        end_classify_time = time.time()

        end_total_time = time.time()
        
        total_spots += len(spots)

        total_time = end_total_time - start_total_time
        crop_time = end_crop_time - start_crop_time
        classify_time = end_classify_time - start_classify_time

        total_time_per_spot = total_time / len(spots)
        crop_time_per_spot = crop_time / len(spots)
        classify_time_per_spot = classify_time / len(spots)

        total_times.append(total_time), total_per_spot.extend([total_time_per_spot] * len(spots))
        crop_images_times.append(crop_time), crop_images_per_spot.extend([crop_time_per_spot] * len(spots))
        classify_images_times.append(classify_time), classify_images_per_spot.extend([classify_time_per_spot] * len(spots))

    return {"totals": (total_times, crop_images_times, classify_images_times), 
        "per_spot": (total_per_spot, crop_images_per_spot, classify_images_per_spot), 
        "images": {"total": len(subset["images"]), "spots": total_spots}}


def execute(model_module, config, args):
    output_name = f"raspberry_{args.base.split('/')[-1]}"
    output_json = {}

    output_json["model"] = config["models"]["module"]
    output_json["dataset"] = config["dataset"]["path"].split("/")[-1]

    print(f"Starting Raspberry Pipeline [BASE: {args.base.split('/')[-1].upper()}][MODEL: {config['models']['module'].split('.')[-1]}]")
    start_global_time = time.time()

    model_config, model_dl_config = config["models"]["config"], config["models"]["dl_config"]

    models = get_models(model_module, args.base, model_config)
    datasets = get_datasets(config)

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize(model_dl_config["img_size"], antialias=True), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    output_json["metrics"] = {}
    dataset_total_times, dataset_crop_images_times, dataset_classify_images_times = [], [], []
    dataset_total_per_spot, dataset_crop_images_per_spot, dataset_classify_images_per_spot = [], [], []
    total_images, total_spots = 0, 0
    for subset, model in models.items():
        print(f"Executing for {subset}...")
        output_json["metrics"][subset] = {}

        subset_start_time = time.time()
        time_artifacts = execute_for_subset(config, model_dl_config, transform, model, datasets[subset])
        subset_end_time = time.time()

        output_json["metrics"][subset]["global"] = subset_end_time - subset_start_time
        output_json["metrics"][subset]["images"] = time_artifacts["images"]

        output_json["metrics"][subset]["total"] = {"average": np.average(time_artifacts["totals"][0]), "std": np.std(time_artifacts["totals"][0])}
        output_json["metrics"][subset]["total_per_spot"] = {"average": np.average(time_artifacts["per_spot"][0]), "std": np.std(time_artifacts["per_spot"][0])}

        output_json["metrics"][subset]["crop"] = {"average": np.average(time_artifacts["totals"][1]), "std": np.std(time_artifacts["totals"][1])}
        output_json["metrics"][subset]["crop_per_spot"] = {"average": np.average(time_artifacts["per_spot"][1]), "std": np.std(time_artifacts["per_spot"][1])}

        output_json["metrics"][subset]["classify"] = {"average": np.average(time_artifacts["totals"][2]), "std": np.std(time_artifacts["totals"][2])}
        output_json["metrics"][subset]["classify_per_spot"] = {"average": np.average(time_artifacts["per_spot"][2]), "std": np.std(time_artifacts["per_spot"][2])}

        total_images += time_artifacts["images"]["total"]
        total_spots += time_artifacts["images"]["spots"]
        dataset_total_times.extend(time_artifacts["totals"][0]), dataset_total_per_spot.extend(time_artifacts["per_spot"][0])
        dataset_crop_images_times.extend(time_artifacts["totals"][1]), dataset_crop_images_per_spot.extend(time_artifacts["per_spot"][1])
        dataset_classify_images_times.extend(time_artifacts["totals"][2]), dataset_classify_images_per_spot.extend(time_artifacts["per_spot"][2])
    
    output_json["metrics"]["average"] = {}
    output_json["metrics"]["average"]["images"] = {"total": total_images, "spots": total_spots}
    output_json["metrics"]["average"]["total"] = {"average": np.average(dataset_total_times), "std": np.std(dataset_total_times)}
    output_json["metrics"]["average"]["total_per_spot"] = {"average": np.average(dataset_total_per_spot), "std": np.std(dataset_total_per_spot)}

    output_json["metrics"]["average"]["crop"] = {"average": np.average(dataset_crop_images_times), "std": np.std(dataset_crop_images_times)}
    output_json["metrics"]["average"]["crop_per_spot"] = {"average": np.average(dataset_crop_images_per_spot), "std": np.std(dataset_crop_images_per_spot)}

    output_json["metrics"]["average"]["classify"] = {"average": np.average(dataset_classify_images_times), "std": np.std(dataset_classify_images_times)}
    output_json["metrics"]["average"]["classify_per_spot"] = {"average": np.average(dataset_classify_images_per_spot), "std": np.std(dataset_classify_images_per_spot)}

    end_global_time = time.time()
    output_json["metrics"]["global"] = end_global_time - start_global_time

    print("Saving result file")
    with open(f"_results/{args.exp}/{output_name}.json", "w") as output:
        json.dump(output_json, output, indent=2)


def main(args):
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    model_module = importlib.import_module(config["models"]["module"])

    execute(model_module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base", "-b", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--exp", "-e", type=str, required=True)

    main(parser.parse_args())