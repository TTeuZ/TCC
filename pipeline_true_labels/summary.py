import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import get_cm
from collections import defaultdict
from datetime import datetime
import numpy as np
import argparse
import json

def print_avg_results(results_by_experiment, file):
    pre_accs, pre_cms, pos_accs, pos_cms = [], [], [], []
    all_images, total_train, total_val = 0, 0, 0

    for result in results_by_experiment[list(results_by_experiment.keys())[0]]:
        all_images += (result["dataset"]["total_train"] + result["dataset"]["total_val"])
        total_train += result["dataset"]["total_train"]
        total_val += result["dataset"]["total_val"]

    for _, results in results_by_experiment.items():
        experiment_pre_cm = sum([get_cm(result["model"]["pre_refinement"]["cm"]) for result in results])
        experiment_pos_cm = sum([get_cm(result["model"]["pos_refinement"]["cm"]) for result in results])

        experiment_pre_acc = np.trace(experiment_pre_cm) / np.sum(experiment_pre_cm)
        experiment_pos_acc = np.trace(experiment_pos_cm) / np.sum(experiment_pos_cm)

        pre_accs.append(experiment_pre_acc), pre_cms.append(experiment_pre_cm)
        pos_accs.append(experiment_pos_acc), pos_cms.append(experiment_pos_cm)

    avg_pre_acc, std_pre_acc = np.average(pre_accs), np.std(pre_accs)
    avg_pos_acc, std_pos_acc = np.average(pos_accs), np.std(pos_accs)

    avg_pre_cms = sum(pre_cms) / len(pre_cms)
    avg_pos_cms = sum(pos_cms) / len(pos_cms)

    file.write(f"AVG of all results:\n")

    file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
    file.write(f"All: {all_images} - Train: {total_train} - Val: {total_val}\n")

    file.write("\n---------------------[MODEL PRE REFINEMENT]--------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_pre_acc:.4f} {std_pre_acc:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_pre_cms[1][1]:6.0f}".ljust(10) + f"{avg_pre_cms[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_pre_cms[0][1]:6.0f}".ljust(10) + f"{avg_pre_cms[0][0]:6.0f}\n")

    file.write("\n---------------------[MODEL POS REFINEMENT]--------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_pos_acc:.4f} {std_pos_acc:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_pos_cms[1][1]:6.0f}".ljust(10) + f"{avg_pos_cms[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_pos_cms[0][1]:6.0f}".ljust(10) + f"{avg_pos_cms[0][0]:6.0f}\n")


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

        file.write(f"Models models module: {config['model']['module']}\n")
        file.write(f"Models models trained with: {config['model']['config']['training_mode']}\n")
        file.write(f"Dataset used: {config['dataset']['path'].split('/')[-1]}\n")
        file.write(f"Training days: {config['dataset']['train_days']}\n")

        file.write("\n###############################################################\n\n")

        print_avg_results(results_by_experiment, file)

        file.write("\n###############################################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    
    main(parser.parse_args())