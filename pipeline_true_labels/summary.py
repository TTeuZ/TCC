import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import get_cm
from collections import defaultdict
from datetime import datetime
import numpy as np
import argparse
import json

def print_results(subset, results, file):
    results.sort(key=lambda result: result["model"]["pos_refinement"]["accuracy"])

    aucs = [result["model"]["metrics"]["auc"] for result in results]
    eers = [result["model"]["metrics"]["eer"] for result in results]
    thresholds = [result["model"]["metrics"]["threshold"] for result in results]
    val_loss = [result["model"]["best_model"]["loss"] for result in results]
    epochs = [result["model"]["best_model"]["epoch_id"] for result in results]
    pre_accs = [result["model"]["pre_refinement"]["accuracy"] for result in results]
    pre_cms = [get_cm(result["model"]["pre_refinement"]["cm"]) for result in results]
    pos_accs = [result["model"]["pos_refinement"]["accuracy"] for result in results]
    pos_cms = [get_cm(result["model"]["pos_refinement"]["cm"]) for result in results]

    avg_auc = sum(aucs) / len(aucs)
    avg_eer = sum(eers) / len(eers)
    avg_threshold = sum(thresholds) / len(thresholds)
    avg_val_loss = sum(val_loss) / len(val_loss)
    avg_epoch = int(sum(epochs) / len(epochs))
    avg_pre_acc = sum(pre_accs) / len(pre_accs)
    avg_pre_cm = np.round(sum(pre_cms) / len(pre_cms))
    avg_pos_acc = sum(pos_accs) / len(pos_accs)
    avg_pos_cm = np.round(sum(pos_cms) / len(pos_cms))

    std_auc = np.std(aucs)
    std_eer = np.std(eers)
    std_threshold = np.std(thresholds)
    std_val_loss = np.std(val_loss)
    std_epoch = np.std(epochs)
    std_pre_acc = np.std(pre_accs)
    std_pos_acc = np.std(pos_accs)

    all_images = results[0]["dataset"]["total_train"] + results[0]["dataset"]["total_val"]
    train_images = results[0]["dataset"]["total_train"]
    val_images = results[0]["dataset"]["total_val"]

    labels_before_leveling_empty = [result["dataset"]["train_labels"]["before_leveling"]["empty"] for result in results]
    labels_before_leveling_occupied = [result["dataset"]["train_labels"]["before_leveling"]["occupied"] for result in results]
    labels_after_leveling_empty = [result["dataset"]["train_labels"]["after_leveling"]["empty"] for result in results]
    labels_after_leveling_occupied = [result["dataset"]["train_labels"]["after_leveling"]["occupied"] for result in results]

    avg_labels_before_leveling_empty = int(np.round(sum(labels_before_leveling_empty) / len(labels_before_leveling_empty)))
    avg_labels_before_leveling_occupied = int(np.round(sum(labels_before_leveling_occupied) / len(labels_before_leveling_occupied)))
    avg_labels_after_leveling_empty = int(np.round(sum(labels_after_leveling_empty) / len(labels_after_leveling_empty)))
    avg_labels_after_leveling_occupied = int(np.round(sum(labels_after_leveling_occupied) / len(labels_after_leveling_occupied)))


    file.write(f"Subset: {subset.split('_')[1]}\n")
    file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
    file.write(f"All: {all_images} - Train: {train_images} - Val: {val_images}\n")
    file.write(f"Train Labels before leveling: {avg_labels_before_leveling_empty}[EMPTY] - {avg_labels_before_leveling_occupied}[OCCUPIED]\n")
    file.write(f"Train Labels before leveling: {avg_labels_after_leveling_empty}[EMPTY] - {avg_labels_after_leveling_occupied}[OCCUPIED]\n")

    file.write("\n-------------------------[ALL RESULTS]-------------------------\n")
    file.write(" - ".join(f"{auc:.4f}" for auc in aucs) + " [AUCS]\n")
    file.write(" - ".join(f"{eer:.4f}" for eer in eers) + " [EERS]\n")
    file.write(" - ".join(f"{threshold:.4f}" for threshold in thresholds) + " [THRESHOLDS]\n")
    file.write(" - ".join(f"{val_loss:.4f}" for val_loss in val_loss) + " [VAL LOSS]\n")
    file.write(" - ".join(f"{epoch:6d}" for epoch in epochs) + " [EPOCHS]\n")
    file.write(" - ".join(f"{acc:.4f}" for acc in pre_accs) + " [PRE ACC]\n")
    file.write(" - ".join(f"{acc:.4f}" for acc in pos_accs) + " [POS ACC]\n")

    file.write("\n-------------------------[AVG RESULT]--------------------------\n")
    file.write(" [AVG]  [STD]".ljust(33) + "Pre Confusion matrix - Rounded\n")
    file.write(f"{avg_auc:.4f} {std_auc:.4f} [AUC]".ljust(47) + f"{avg_pre_cm[1][1]:6.0f}".ljust(10) + f"{avg_pre_cm[1][0]:6.0f}\n")
    file.write(f"{avg_eer:.4f} {std_eer:.4f} [EER]".ljust(47) + f"{avg_pre_cm[0][1]:6.0f}".ljust(10) + f"{avg_pre_cm[0][0]:6.0f}\n")
    file.write(f"{avg_threshold:.4f} {std_threshold:.4f} [THRESHOLD]\n")
    file.write(f"{avg_val_loss:.4f} {std_val_loss:.4f} [VAL LOSS]".ljust(33) + "Pos Confusion matrix - Rounded\n")
    file.write(f"{avg_epoch:6d} {std_epoch:.4f} [EPOCH]".ljust(47) + f"{avg_pos_cm[1][1]:6.0f}".ljust(10) + f"{avg_pos_cm[1][0]:6.0f}\n")
    file.write(f"{avg_pre_acc:.4f} {std_pre_acc:.4f} [PRE ACC]".ljust(47) + f"{avg_pos_cm[0][1]:6.0f}".ljust(10) + f"{avg_pos_cm[0][0]:6.0f}\n")
    file.write(f"{avg_pos_acc:.4f} {std_pos_acc:.4f} [POS ACC]\n\n")


def print_avg_results(results_by_subset, file):
    pre_accs, pre_cms, pos_accs, pos_cms = [], [], [], []
    all_images, total_train, total_val = 0, 0, 0
    labels_before_leveling, labels_after_leveling = {"empty": [], "occupied": []}, {"empty": [], "occupied": []}

    for subset, results in results_by_subset.items():
        all_images += (results_by_subset[subset][0]["dataset"]["total_train"] + results_by_subset[subset][0]["dataset"]["total_val"])
        total_train += results_by_subset[subset][0]["dataset"]["total_train"]
        total_val += results_by_subset[subset][0]["dataset"]["total_val"]
        subset_pre_cms, subset_pos_cms = [], []
        subset_used_images = []
        subset_labels_before_leveling, subset_labels_after_leveling = {"empty": [], "occupied": []}, {"empty": [], "occupied": []}

        for result in results:
            pre_accs.append(result["model"]["pre_refinement"]["accuracy"])
            pos_accs.append(result["model"]["pos_refinement"]["accuracy"])

            subset_pre_cms.append(get_cm(result["model"]["pre_refinement"]["cm"]))
            subset_pos_cms.append(get_cm(result["model"]["pos_refinement"]["cm"]))

            subset_labels_before_leveling["empty"].append(result["dataset"]["train_labels"]["before_leveling"]["empty"])
            subset_labels_before_leveling["occupied"].append(result["dataset"]["train_labels"]["before_leveling"]["occupied"])
            subset_labels_after_leveling["empty"].append(result["dataset"]["train_labels"]["after_leveling"]["empty"])
            subset_labels_after_leveling["occupied"].append(result["dataset"]["train_labels"]["after_leveling"]["occupied"])

        subset_pre_cms = np.round(sum(subset_pre_cms) / len(subset_pre_cms))
        subset_pos_cms = np.round(sum(subset_pos_cms) / len(subset_pos_cms))

        subset_labels_before_leveling["empty"] = int(np.round(sum(subset_labels_before_leveling["empty"]) / len(subset_labels_before_leveling["empty"])))
        subset_labels_before_leveling["occupied"] = int(np.round(sum(subset_labels_before_leveling["occupied"]) / len(subset_labels_before_leveling["occupied"])))
        subset_labels_after_leveling["empty"] = int(np.round(sum(subset_labels_after_leveling["empty"]) / len(subset_labels_after_leveling["empty"])))
        subset_labels_after_leveling["occupied"] = int(np.round(sum(subset_labels_after_leveling["occupied"]) / len(subset_labels_after_leveling["occupied"])))

        pre_cms.append(subset_pre_cms)
        pos_cms.append(subset_pos_cms)

        labels_before_leveling["empty"].append(subset_labels_before_leveling["empty"])
        labels_before_leveling["occupied"].append(subset_labels_before_leveling["occupied"])
        labels_after_leveling["empty"].append(subset_labels_after_leveling["empty"])
        labels_after_leveling["occupied"].append(subset_labels_after_leveling["occupied"])

    avg_pre_accs = sum(pre_accs) / len(pre_accs)
    avg_pos_accs = sum(pos_accs) / len(pos_accs)

    std_pre_accs = np.std(pre_accs)
    std_pos_accs = np.std(pos_accs)

    avg_pre_cms = sum(pre_cms)
    avg_pos_cms = sum(pos_cms)

    avg_labels_before_leveling = {"empty": sum(labels_before_leveling["empty"]), "occupied": sum(labels_before_leveling["occupied"])}
    avg_labels_after_leveling = {"empty": sum(labels_after_leveling["empty"]), "occupied": sum(labels_after_leveling["occupied"])}

    file.write(f"AVG of all results:\n")

    file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
    file.write(f"All: {all_images} - Train: {total_train} - Val: {total_val}\n")
    file.write(f"Train Labels before leveling: {avg_labels_before_leveling['empty']}[EMPTY] - {avg_labels_before_leveling['occupied']}[OCCUPIED]\n")
    file.write(f"Train Labels before leveling: {avg_labels_after_leveling['empty']}[EMPTY] - {avg_labels_after_leveling['occupied']}[OCCUPIED]\n")

    file.write("\n---------------------[MODEL PRE REFINEMENT]--------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_pre_accs:.4f} {std_pre_accs:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_pre_cms[1][1]:6.0f}".ljust(10) + f"{avg_pre_cms[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_pre_cms[0][1]:6.0f}".ljust(10) + f"{avg_pre_cms[0][0]:6.0f}\n")

    file.write("\n---------------------[MODEL POS REFINEMENT]--------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_pos_accs:.4f} {std_pos_accs:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_pos_cms[1][1]:6.0f}".ljust(10) + f"{avg_pos_cms[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_pos_cms[0][1]:6.0f}".ljust(10) + f"{avg_pos_cms[0][0]:6.0f}\n")


def get_results_by_subset(path):
    models_results = [folder for folder in os.listdir(path) if "model" in folder]

    results_by_subset = defaultdict(list)
    for model in models_results:
        jsons = os.listdir(f"{path}/{model}")
        results = [json.load(open(f"{path}/{model}/{file}", "r")) for file in jsons]
        for result in results:
            results_by_subset[f"model_{result['dataset']['subset']}"].append(result)

    results_by_subset = dict(results_by_subset)
    return dict(sorted(results_by_subset.items()))


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

        file.write(f"Models models module: {config['model']['module']}\n")
        file.write(f"Models models trained with: {config['model']['config']['training_mode']}\n")
        file.write(f"Dataset used: {config['dataset']['path']}\n")
        file.write(f"Training days: {config['dataset']['train_days']}\n")

        file.write("\n###############################################################\n\n")

        print_avg_results(results_by_subset, file)
        file.write("\n###############################################################\n\n")
        for subset, results in results_by_subset.items():
            print_results(subset, results, file)
            file.write("###############################################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    
    main(parser.parse_args())