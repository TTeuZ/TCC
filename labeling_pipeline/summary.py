import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import get_cm
from datetime import datetime
import numpy as np
import argparse
import json

def print_results(subset, results, file):
    results.sort(key=lambda result: result["model"]["test"]["accuracy"])

    aucs = [result["model"]["metrics"]["auc"] for result in results]
    eers = [result["model"]["metrics"]["eer"] for result in results]
    thresholds = [result["model"]["metrics"]["threshold"] for result in results]
    val_loss = [result["model"]["best_model"]["loss"] for result in results]
    epochs = [result["model"]["best_model"]["epoch_id"] for result in results]
    accuracies = [result["model"]["test"]["accuracy"] for result in results]
    cms = [get_cm(result["model"]["test"]["cm"]) for result in results]

    avg_auc = sum(aucs) / len(aucs)
    avg_eer = sum(eers) / len(eers)
    avg_threshold = sum(thresholds) / len(thresholds)
    avg_val_loss = sum(val_loss) / len(val_loss)
    avg_epoch = int(sum(epochs) / len(epochs))
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_cm = np.round(sum(cms) / len(cms))

    std_auc = np.std(aucs)
    std_eer = np.std(eers)
    std_threshold = np.std(thresholds)
    std_val_loss = np.std(val_loss)
    std_epoch = np.std(epochs)
    std_accuracy = np.std(accuracies)

    all_images = results[0]["dataset"]["classify"]["all_images"]
    used_images = [result["dataset"]["classify"]["used_images"] for result in results]
    avg_used_images = int(np.round(sum(used_images) / len(used_images)))
    std_used_images = np.std(used_images)
    pct_used_images = avg_used_images / all_images

    file.write(f"Subset: {subset.split('_')[1]}\n")
    file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
    file.write(f"All: {all_images} - Used: {avg_used_images}[AVG] - {std_used_images:.2f}[STD] - {pct_used_images:.2f}[%]\n")

    file.write("\n-------------------------[ALL RESULTS]-------------------------\n")
    file.write(" - ".join(f"{auc:.4f}" for auc in aucs) + " [AUCS]\n")
    file.write(" - ".join(f"{eer:.4f}" for eer in eers) + " [EERS]\n")
    file.write(" - ".join(f"{threshold:.4f}" for threshold in thresholds) + " [THRESHOLDS]\n")
    file.write(" - ".join(f"{val_loss:.4f}" for val_loss in val_loss) + " [VAL LOSS]\n")
    file.write(" - ".join(f"{epoch:6d}" for epoch in epochs) + " [EPOCHS]\n")
    file.write(" - ".join(f"{acc:.4f}" for acc in accuracies) + " [ACCURACY]\n")

    file.write("\n-------------------------[AVG RESULT]--------------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_auc:.4f} {std_auc:.4f} [AUC]".ljust(47) + f"{avg_cm[1][1]:6.0f}".ljust(10) + f"{avg_cm[1][0]:6.0f}\n")
    file.write(f"{avg_eer:.4f} {std_eer:.4f} [EER]".ljust(47) + f"{avg_cm[0][1]:6.0f}".ljust(10) + f"{avg_cm[0][0]:6.0f}\n")
    file.write(f"{avg_threshold:.4f} {std_threshold:.4f} [THRESHOLD]\n")
    file.write(f"{avg_val_loss:.4f} {std_val_loss:.4f} [VAL LOSS]\n")
    file.write(f"{avg_epoch:6d} {std_epoch:.4f} [EPOCH]\n")
    file.write(f"{avg_accuracy:.4f} {std_accuracy:.4f} [ACCURACY]\n")

    file.write("\n------------------------[FATHER MODEL]-------------------------\n")
    father_accuracies = [result["father_model"]["test"]["accuracy"] for result in results]
    father_cms = [get_cm(result["father_model"]["test"]["cm"]) for result in results]

    father_avg_accuracy = sum(father_accuracies) / len(father_accuracies)
    father_avg_cm = np.round(sum(father_cms) / len(father_cms))
    father_std_accuracy = np.std(father_accuracies)

    file.write(" - ".join(f"{acc:.4f}" for acc in father_accuracies) + " [ACCURACY]\n\n")

    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{father_avg_accuracy:.4f} {father_std_accuracy:.4f} [ACCURACY]".ljust(47))

    file.write(f"{father_avg_cm[1][1]:6.0f}".ljust(10) + f"{father_avg_cm[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{father_avg_cm[0][1]:6.0f}".ljust(10) + f"{father_avg_cm[0][0]:6.0f}\n\n")


def print_avg_results(results_by_subset, file):
    accuracies, cms, father_accuracies, father_cms = [], [], [], []
    all_images, used_images = 0, []
    
    for subset in results_by_subset:
        subset_cms, subset_father_cms = [], []

        all_images += results_by_subset[subset][0]["dataset"]["classify"]["all_images"]
        subset_used_images = []

        for result in results_by_subset[subset]:
            accuracies.append(result["model"]["test"]["accuracy"])
            subset_cms.append(get_cm(result["model"]["test"]["cm"]))
            father_accuracies.append(result["father_model"]["test"]["accuracy"])
            subset_father_cms.append(get_cm(result["father_model"]["test"]["cm"]))

            subset_used_images.append(result["dataset"]["classify"]["used_images"])
        
        subset_cms = np.round(sum(subset_cms) / len(subset_cms))
        cms.append(subset_cms)

        subset_father_cms = np.round(sum(subset_father_cms) / len(subset_father_cms))
        father_cms.append(subset_father_cms)

        subset_used_images = int(np.round(sum(subset_used_images) / len(subset_used_images)))
        used_images.append(subset_used_images)
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_cm = sum(cms)
    std_accuracy = np.std(accuracies)

    father_avg_accuracy = sum(father_accuracies) / len(father_accuracies)
    father_avg_cm = sum(father_cms)
    father_std_accuracy = np.std(father_accuracies)

    avg_used_images = sum(used_images)
    pct_used_images = avg_used_images / all_images
    
    file.write(f"AVG of all results:\n")

    file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
    file.write(f"All: {all_images} - Used: {avg_used_images}[AVG] - {pct_used_images:.2f}[%]\n")

    file.write("\n---------------------------[MODEL]-----------------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_accuracy:.4f} {std_accuracy:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_cm[1][1]:6.0f}".ljust(10) + f"{avg_cm[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_cm[0][1]:6.0f}".ljust(10) + f"{avg_cm[0][0]:6.0f}\n\n")

    file.write("\n------------------------[FATHER MODEL]-------------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{father_avg_accuracy:.4f} {father_std_accuracy:.4f} [ACCURACY]".ljust(47))

    file.write(f"{father_avg_cm[1][1]:6.0f}".ljust(10) + f"{father_avg_cm[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{father_avg_cm[0][1]:6.0f}".ljust(10) + f"{father_avg_cm[0][0]:6.0f}\n")


def get_results_by_subset(path):
    models_results = [folder for folder in os.listdir(path) if "model" in folder]

    subsets = set()
    for subset in os.listdir(f"{path}/{models_results[0]}"):
        subsets.add(subset.split(".")[0])
    subsets = sorted(subsets)

    results_by_subset = {subset: [] for subset in subsets}
    for model in models_results:
        jsons = os.listdir(f"{path}/{model}")
        results = [json.load(open(f"{path}/{model}/{file}", "r")) for file in jsons]
        for result in results:
            results_by_subset[f"model_{result['dataset']['subset']}"].append(result)

    return results_by_subset


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    results_by_subset = get_results_by_subset(args.files)

    begin_date = datetime.strptime(args.begin_date, '%Y-%m-%d--%H:%M:%S')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d--%H:%M:%S')
    time_spent = end_date - begin_date

    summary_name = args.files.split("/")[1]
    with open(f"_summaries/{summary_name}.txt", "w") as file:
        file.write(f"Begin: {args.begin_date} End: {args.end_date} -- Time spent: {time_spent} \n\n")
        
        file.write(f"Details in {args.files}")
        file.write("\n---------------------------------------------------------------\n\n")

        file.write(f"Father models trained with: {config['fathers']['trained_at']}\n")
        file.write(f"Father models module: {config['fathers']['module']}\n")
        file.write(f"Son models module: {config['model']['module']}\n")
        file.write(f"Dataset used: {config['dataset']['path']}\n")

        file.write("\n###############################################################\n\n")

        print_avg_results(results_by_subset, file)

        file.write("\n###############################################################\n\n")
        for subset in results_by_subset:
            print_results(subset, results_by_subset[subset], file)
            file.write("###############################################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    
    main(parser.parse_args())