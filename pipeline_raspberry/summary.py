import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import argparse
import json

def print_avg_results(experiments, file):
    global_time = [result["metrics"]["global"] for result in experiments]

    all_images = experiments[0]["metrics"]["average"]["images"]["total"]
    all_spots = experiments[0]["metrics"]["average"]["images"]["spots"]

    dataset_total = [result["metrics"]["average"]["total"]["average"] for result in experiments]
    dataset_total_per_spot = [result["metrics"]["average"]["total_per_spot"]["average"] for result in experiments]
    dataset_crop = [result["metrics"]["average"]["crop"]["average"] for result in experiments]
    dataset_crop_per_spot = [result["metrics"]["average"]["crop_per_spot"]["average"] for result in experiments]
    dataset_classify = [result["metrics"]["average"]["classify"]["average"] for result in experiments]
    dataset_classify_per_spot = [result["metrics"]["average"]["classify_per_spot"]["average"] for result in experiments]

    avg_global_time, std_global_time = np.average(global_time), np.std(global_time)

    avg_dataset_total, std_dataset_total = np.average(dataset_total), np.std(dataset_total)
    avg_dataset_total_per_spot, std_dataset_total_per_spot = np.average(dataset_total_per_spot), np.std(dataset_total_per_spot)
    avg_dataset_crop, std_dataset_crop = np.average(dataset_crop), np.std(dataset_crop)
    avg_dataset_crop_per_spot, std_dataset_crop_per_spot = np.average(dataset_crop_per_spot), np.std(dataset_crop_per_spot)
    avg_dataset_classify, std_dataset_classify = np.average(dataset_classify), np.std(dataset_classify)
    avg_dataset_classify_per_spot, std_dataset_classify_per_spot = np.average(dataset_classify_per_spot), np.std(dataset_classify_per_spot)

    file.write("Dataset results:\n\n")
    file.write(f"Global time: {avg_global_time:.4f}[AVG] - {std_global_time:.4f}[STD]\n")
    file.write(f"All images: {all_images} - All spots: {all_spots}\n\n")
    
    file.write("Total process (crop + classify):\n")
    file.write(f"Per image: {avg_dataset_total:.4f}[AVG] - {std_dataset_total:.4f}[STD]\n")
    file.write(f"Per spot: {avg_dataset_total_per_spot:.4f}[AVG] - {std_dataset_total_per_spot:.4f}[STD]\n\n")

    file.write("Crop process:\n")
    file.write(f"Total: {avg_dataset_crop:.4f}[AVG] - {std_dataset_crop:.4f}[STD]\n")
    file.write(f"Per spot: {avg_dataset_crop_per_spot:.4f}[AVG] - {std_dataset_crop_per_spot:.4f}[STD]\n\n")

    file.write("Classify process:\n")
    file.write(f"Total: {avg_dataset_classify:.4f}[AVG] - {std_dataset_classify:.4f}[STD]\n")
    file.write(f"Per spot: {avg_dataset_classify_per_spot:.4f}[AVG] - {std_dataset_classify_per_spot:.4f}[STD]\n\n")

    file.write("\n###############################################################\n\n")

    subsets = [value for value in experiments[0]["metrics"].keys() if value not in ["average", "global"]]
    for subset in subsets:
        subset_global_time = [result["metrics"][subset]["global"] for result in experiments]
        images = experiments[0]["metrics"][subset]["images"]["total"]
        spots = experiments[0]["metrics"][subset]["images"]["spots"]

        subset_total = [result["metrics"][subset]["total"]["average"] for result in experiments]
        subset_total_per_spot = [result["metrics"][subset]["total_per_spot"]["average"] for result in experiments]
        subset_crop = [result["metrics"][subset]["crop"]["average"] for result in experiments]
        subset_crop_per_spot = [result["metrics"][subset]["crop_per_spot"]["average"] for result in experiments]
        subset_classify = [result["metrics"]["average"]["classify"]["average"] for result in experiments]
        subset_classify_per_spot = [result["metrics"]["average"]["classify_per_spot"]["average"] for result in experiments]

        avg_subset_global_time, std_subset_global_time = np.average(subset_global_time), np.std(subset_global_time)

        avg_subset_total, std_subset_total = np.average(subset_total), np.std(subset_total)
        avg_subset_total_per_spot, std_subset_total_per_spot = np.average(subset_total_per_spot), np.std(subset_total_per_spot)
        avg_subset_crop, std_subset_crop = np.average(subset_crop), np.std(subset_crop)
        avg_subset_crop_per_spot, std_subset_crop_per_spot = np.average(subset_crop_per_spot), np.std(subset_crop_per_spot)
        avg_subset_classify, std_subset_classify = np.average(subset_classify), np.std(subset_classify)
        avg_subset_classify_per_spot, std_subset_classify_per_spot = np.average(subset_classify_per_spot), np.std(subset_classify_per_spot)
        
        file.write(f"{subset} results:\n\n")
        file.write(f"Global time: {avg_subset_global_time:.4f}[AVG] - {std_subset_global_time:.4f}[STD]\n")
        file.write(f"All images: {images} - All spots: {spots}\n\n")

        file.write("Total process (crop + classify):\n")
        file.write(f"Per image: {avg_subset_total:.4f}[AVG] - {std_subset_total:.4f}[STD]\n")
        file.write(f"Per spot: {avg_subset_total_per_spot:.4f}[AVG] - {std_subset_total_per_spot:.4f}[STD]\n\n")
    
        file.write("Crop process:\n")
        file.write(f"Total: {avg_subset_crop:.4f}[AVG] - {std_subset_crop:.4f}[STD]\n")
        file.write(f"Per spot: {avg_subset_crop_per_spot:.4f}[AVG] - {std_subset_crop_per_spot:.4f}[STD]\n\n")
    
        file.write("Classify process:\n")
        file.write(f"Total: {avg_subset_classify:.4f}[AVG] - {std_subset_classify:.4f}[STD]\n")
        file.write(f"Per spot: {avg_subset_classify_per_spot:.4f}[AVG] - {std_subset_classify_per_spot:.4f}[STD]\n")

        file.write("\n###############################################################\n\n")


def get_experiments(path):
    jsons = [file for file in os.listdir(path) if ".json" in file]
    results = [json.load(open(f"{path}/{file}", "r")) for file in jsons]

    return results


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"

    summary_name = args.files.split("/")[1]
    config = json.load(open(args.config, "r"))
    experiments = get_experiments(args.files)

    with open(f"_summaries/{summary_name}.txt", "w") as file:
        file.write(f"Details in {args.files}\n")
        file.write("\n###############################################################\n\n")

        file.write(f"Model: {config['models']['module']}\n")
        file.write(f"Dataset: {config['dataset']['path'].split('/')[-1]}\n")
        file.write(f"Exclude days: {config['dataset']['exclude_days']}\n\n")

        file.write("###############################################################\n\n")

        print_avg_results(experiments, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    
    main(parser.parse_args())