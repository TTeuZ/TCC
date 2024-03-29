from utils.utils import get_cm
import numpy as np
import argparse
import json
import os

def print_results(results, file):
    train_ds = results[0]["dataset"]["train"]
    test_ds = results[0]["dataset"]["test"]
    results.sort(key=lambda result: result["test"]["average"]["accuracy"])

    aucs = [result["metrics"]["auc"] for result in results]
    eers = [result["metrics"]["eer"] for result in results]
    thresholds = [result["metrics"]["threshold"] for result in results]
    val_loss = [result["best_model"]["loss"] for result in results]
    epochs = [result["best_model"]["epoch_id"] for result in results]
    accuracies = [result["test"]["average"]["accuracy"] for result in results]
    cms = [get_cm(result["test"]["average"]["cm"]) for result in results]

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

    file.write(f"Train: {train_ds} - Test: {test_ds}\n")
    file.write("\n-------------------------[ALL RESULTS]-------------------------\n")
    file.write(" - ".join(f"{auc:.4f}" for auc in aucs) + " [AUCS]\n")
    file.write(" - ".join(f"{eer:.4f}" for eer in eers) + " [EERS]\n")
    file.write(" - ".join(f"{threshold:.4f}" for threshold in thresholds) + " [THRESHOLDS]\n")
    file.write(" - ".join(f"{val_loss:.4f}" for val_loss in val_loss) + " [VAL LOSS]\n")
    file.write(" - ".join(f"{epoch:6d}" for epoch in epochs) + " [EPOCHS]\n")
    file.write(" - ".join(f"{acc:.4f}" for acc in accuracies) + " [ACCURACY]\n")

    file.write("\n-------------------------[AVG RESULT]--------------------------\n")
    file.write(" [AVG]  [STD] \t\t\t\t\t  Confusion matrix - Rounded\n")
    file.write(f"{avg_auc:.4f} {std_auc:.4f} [AUC] \t\t\t\t\t  {avg_cm[1][1]:6.0f} \t  {avg_cm[1][0]:6.0f}\n")
    file.write(f"{avg_eer:.4f} {std_eer:.4f} [EER] \t\t\t\t\t  {avg_cm[0][1]:6.0f} \t  {avg_cm[0][0]:6.0f}\n")
    file.write(f"{avg_threshold:.4f} {std_threshold:.4f} [THRESHOLD]\n")
    file.write(f"{avg_val_loss:.4f} {std_val_loss:.4f} [VAL LOSS]\n")
    file.write(f"{avg_epoch:6d} {std_epoch:.4f} [EPOCH]\n")
    file.write(f"{avg_accuracy:.4f} {std_accuracy:.4f} [ACCURACY]\n")

    file.write("\n-------------------------[ BY SUBSET ]-------------------------\n")
    subsets = set()
    for subset in results[0]["test"]["subsets"]:
        subsets.add(subset)
    
    results_by_subset = {subset: {'accuracy': 0.0, 'std_accuracy': [], 'cm': [[0, 0], [0, 0]]} for subset in subsets}
    results_by_subset = dict(sorted(results_by_subset.items()))

    for result in results:
        for subset in result["test"]["subsets"]:
            results_by_subset[subset]["accuracy"] += result["test"]["subsets"][subset]["accuracy"]
            results_by_subset[subset]["std_accuracy"].append(result["test"]["subsets"][subset]["accuracy"])
            results_by_subset[subset]["cm"] += get_cm(result["test"]["subsets"][subset]["cm"])

    for subset in results_by_subset:
        results_by_subset[subset]["accuracy"] = results_by_subset[subset]["accuracy"] / len(accuracies) # Any array would work
        results_by_subset[subset]["std_accuracy"] = np.std(results_by_subset[subset]["std_accuracy"])
        results_by_subset[subset]["cm"] = np.round(results_by_subset[subset]["cm"] / len(cms))
    
    for subset in results_by_subset:
        accuracy = results_by_subset[subset]["accuracy"]
        std_accuracy = results_by_subset[subset]["std_accuracy"]
        cm = results_by_subset[subset]["cm"]

        file.write(f"{subset}:".ljust(37))
        file.write("Confusion matrix - Rounded\n")

        file.write(f"{accuracy:.4f} [ACCURACY]".ljust(47))
        file.write(f"{cm[1][1]:6.0f} \t  {cm[1][0]:6.0f}\n")
        file.write(f"{std_accuracy:.4f} [STD_ACCURACY]".ljust(47))
        file.write(f"{cm[0][1]:6.0f} \t  {cm[0][0]:6.0f}\n")
        file.write("\n")


def get_experiments(path):
    jsons = [file for file in os.listdir(path) if ".json" in file]
    results = [json.load(open(f"{path}/{file}", "r")) for file in jsons]

    types = set()
    for result in results:
        types.add(result["dataset"]["train"])
    
    experiments = {experiment: [] for experiment in types}
    for result in results:
        experiments[result["dataset"]["train"]].append(result)

    return experiments


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"

    experiments = get_experiments(args.files)
    summary_name = args.files.split("/")[1]

    with open(f"_summaries/{summary_name}.txt", "w") as file:
        file.write(f"Model: {args.model} \n")
        file.write(f"Loss: {args.loss} \n")
        file.write(f"Split: {args.split} \n\n")
        file.write("###############################################################\n\n")

        for experiment in experiments:
            print_results(experiments[experiment], file)
            file.write("###############################################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--loss", "-l", type=str, required=True)
    parser.add_argument("--split", "-s", type=float, required=True)

    main(parser.parse_args())