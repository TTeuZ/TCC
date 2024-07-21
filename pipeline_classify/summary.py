import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import get_cm
from collections import defaultdict
from datetime import datetime
import numpy as np
import argparse
import json

def print_avg_results(results_by_experiment, file):
    all_images, used_images, wrong_labeled = 0, defaultdict(list), defaultdict(list)
    father_accs, pre_accs, pos_accs = [], [], defaultdict(list)

    dummy_results = results_by_experiment[list(results_by_experiment.keys())[0]]
    thresholds = [value for value in dummy_results[0]["model"].keys() if "classify" in value]

    for result in results_by_experiment[list(results_by_experiment.keys())[0]]:
        all_images += result["model"][thresholds[0]]["all_images"]

    for _, results in results_by_experiment.items():
        experiment_father_cm = sum([get_cm(result["father_model"]["test"]["cm"]) for result in results])
        experiment_pre_cm = sum([get_cm(result["model"]["pre_refinement"]["cm"]) for result in results])

        experiment_father_acc = np.trace(experiment_father_cm) / np.sum(experiment_father_cm)
        experiment_pre_acc = np.trace(experiment_pre_cm) / np.sum(experiment_pre_cm)

        experiment_used_images, experiment_wrong_labeled = dict(), dict()
        experiment_pos_cms, experiment_pos_accs = dict(), dict()
        for threshold in thresholds:
            experiment_used_images[threshold] = sum([result["model"][threshold]["used_images"] for result in results])
            experiment_wrong_labeled[threshold] = sum([result["model"][threshold]["wrong_labels"] for result in results])
            experiment_pos_cms[threshold] = sum([get_cm(result["model"][threshold]["pos_refinement"]["cm"]) for result in results])
            experiment_pos_accs[threshold] = np.trace(experiment_pos_cms[threshold]) / np.sum(experiment_pos_cms[threshold])

        father_accs.append(experiment_father_acc)
        pre_accs.append(experiment_pre_acc)

        for threshold in thresholds:
            used_images[threshold].append(experiment_used_images[threshold])
            wrong_labeled[threshold].append(experiment_wrong_labeled[threshold])
            pos_accs[threshold].append(experiment_pos_accs[threshold])

    avg_father_acc, std_father_acc = np.average(father_accs), np.std(father_accs)
    avg_pre_acc, std_pre_acc = np.average(pre_accs), np.std(pre_accs)

    avg_used_images, std_used_images = defaultdict(float), defaultdict(float)
    avg_wrong_labeled, std_wrong_labeled = defaultdict(float), defaultdict(float)
    avg_pos_accs, std_pos_accs = defaultdict(float), defaultdict(float)
    for threshold in thresholds:
        avg_used_images[threshold] = np.average(used_images[threshold])
        std_used_images[threshold] = np.std(used_images[threshold])

        avg_wrong_labeled[threshold] = np.average(wrong_labeled[threshold])
        std_wrong_labeled[threshold] = np.std(wrong_labeled[threshold])

        avg_pos_accs[threshold] = np.average(pos_accs[threshold])
        std_pos_accs[threshold] = np.std(pos_accs[threshold])

    file.write(f"AVG of all results:\n")

    file.write("\n------------------------[FATHER MODEL]-------------------------\n")
    file.write(" [AVG]  [STD]\n")
    file.write(f"{avg_father_acc:.4f} {std_father_acc:.4f} [ACCURACY]\n")

    file.write("\n-------------------[SON MODEL PRE REFINEMENT]------------------\n")
    file.write(" [AVG]  [STD]\n")
    file.write(f"{avg_pre_acc:.4f} {std_pre_acc:.4f} [ACCURACY]\n")

    file.write("\n-----------------------[CLASSIFY CHECK]------------------------\n")

    file.write("  [ACCURACY]".ljust(21) + "[USED]".ljust(15) + "[WRONG]\n")

    for threshold in thresholds:
        file.write(f"{avg_pos_accs[threshold]:.4f} ({std_pos_accs[threshold]:.4f})".ljust(18))
        file.write(f"{avg_used_images[threshold]:.0f} ({std_used_images[threshold]:.0f})".ljust(15))
        file.write(f"{avg_wrong_labeled[threshold]:.0f} ({std_wrong_labeled[threshold]:.2f})".ljust(16))
        file.write(f"[{threshold.upper()}]\n")


def get_results_by_experiment(path):
    models_results = [folder for folder in os.listdir(path) if "model" in folder]
    
    results_by_experiment = dict()
    for model in models_results:
        jsons = sorted(os.listdir(f"{path}/{model}"))
        results = [json.load(open(f"{path}/{model}/{file}", "r")) for file in jsons]
        results_by_experiment[model] = results

    return results_by_experiment


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    results_by_experiment = get_results_by_experiment(args.files)

    begin_date = datetime.strptime(args.begin_date, '%Y-%m-%d--%H:%M:%S')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d--%H:%M:%S')
    time_spent = end_date - begin_date

    summary_name = args.files.split("/")[1]
    with open(f"_summaries/{summary_name}.txt", "w") as file:
        file.write(f"Begin: {args.begin_date} End: {args.end_date} -- Time spent: {time_spent} \n\n")

        file.write(f"Details in {args.files}")
        file.write("\n---------------------------------------------------------------\n\n")

        file.write(f"Father models trained with: {config['fathers']['trained_at'].split('/')[-1]}\n")
        file.write(f"Father models module: {config['fathers']['module']}\n")
        file.write(f"Father models trained with: {config['fathers']['config']['training_mode']}\n")
        file.write(f"Son models module: {config['model']['module']}\n")
        file.write(f"Son models trained with: {config['model']['config']['training_mode']}\n")
        file.write(f"Dataset used: {config['dataset']['path'].split('/')[-1]}\n")
        file.write(f"Training days: {config['dataset']['train_days']}\n")
        file.write(f"Split: {config['dataset']['split']}\n")

        file.write("\n###############################################################\n\n")

        print_avg_results(results_by_experiment, file)

        file.write("\n###############################################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    
    main(parser.parse_args())