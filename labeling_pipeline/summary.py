import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import get_cm
from datetime import datetime
import numpy as np
import argparse
import json

def print_avg_results(results_by_experiment, file):
    pre_accs, pre_cms, pos_accs, pos_cms, father_accs, father_cms = [], [], [], [], [], []
    all_images, used_images, wrong_labeled, empty_wrong, occupied_wrong = 0, [], [], [], []

    for result in results_by_experiment[list(results_by_experiment.keys())[0]]:
        all_images += result["dataset"]["classify"]["all_images"]

    for _, results in results_by_experiment.items():
        experiment_pre_cm = sum([get_cm(result["model"]["pre_refinement"]["cm"]) for result in results])
        experiment_pos_cm = sum([get_cm(result["model"]["pos_refinement"]["cm"]) for result in results])
        experiment_father_cm = sum([get_cm(result["father_model"]["test"]["cm"]) for result in results])

        experiment_pre_acc = np.trace(experiment_pre_cm) / np.sum(experiment_pre_cm)
        experiment_pos_acc = np.trace(experiment_pos_cm) / np.sum(experiment_pos_cm)
        experiment_father_acc = np.trace(experiment_father_cm) / np.sum(experiment_father_cm)

        experiment_used_images = sum([result["dataset"]["classify"]["used_images"] for result in results])
        experiment_wrong_labeled = sum([result["dataset"]["classify"]["wrong_labels"] for result in results])
        experiment_empty_wrong = sum([result["dataset"]["classify"]["empty_wrong"] for result in results])
        experiment_occupied_wrong = sum([result["dataset"]["classify"]["occupied_wrong"] for result in results])

        pre_accs.append(experiment_pre_acc), pre_cms.append(experiment_pre_cm)
        pos_accs.append(experiment_pos_acc), pos_cms.append(experiment_pos_cm)
        father_accs.append(experiment_father_acc), father_cms.append(experiment_father_cm)

        used_images.append(experiment_used_images)
        wrong_labeled.append(experiment_wrong_labeled)
        empty_wrong.append(experiment_empty_wrong)
        occupied_wrong.append(experiment_occupied_wrong)

    avg_pre_acc, std_pre_acc = np.average(pre_accs), np.std(pre_accs)
    avg_pos_acc, std_pos_acc = np.average(pos_accs), np.std(pos_accs)
    avg_father_acc, std_father_acc = np.average(father_accs), np.std(father_accs)

    avg_pre_cms = sum(pre_cms) / len(pre_cms)
    avg_pos_cms = sum(pos_cms) / len(pos_cms)
    avg_father_cms = sum(father_cms) / len(father_cms)

    avg_used_images, std_used_images = np.average(used_images), np.std(used_images)
    avg_wrong_labeled, std_wrong_labeled = np.average(wrong_labeled), np.std(wrong_labeled)
    avg_empty_wrong, std_empty_wrong = np.average(empty_wrong), np.std(empty_wrong)
    avg_occupied_wrong, std_occupied_wrong = np.average(occupied_wrong), np.std(occupied_wrong)

    pct_used_images = avg_used_images / all_images;

    file.write(f"AVG of all results:\n")

    file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
    file.write(f"All: {all_images} - Used: {avg_used_images}[AVG] | {std_used_images:.2f}[STD] - {pct_used_images:.2f}[%]\n")
    file.write(f"All wrong: {avg_wrong_labeled}[AVG] - {std_wrong_labeled:.2f}[STD]\n")
    file.write(f"Empty wrong: {avg_empty_wrong}[AVG] - {std_empty_wrong:.2f}[STD]\n")
    file.write(f"Occupied wrong: {avg_occupied_wrong}[AVG] - {std_occupied_wrong:.2f}[STD]\n")

    file.write("\n------------------------[FATHER MODEL]-------------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_father_acc:.4f} {std_father_acc:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_father_cms[1][1]:6.0f}".ljust(10) + f"{avg_father_cms[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_father_cms[0][1]:6.0f}".ljust(10) + f"{avg_father_cms[0][0]:6.0f}\n")

    file.write("\n-------------------[SON MODEL PRE REFINEMENT]------------------\n")
    file.write(" [AVG]  [STD]".ljust(37) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_pre_acc:.4f} {std_pre_acc:.4f} [ACCURACY]".ljust(47))

    file.write(f"{avg_pre_cms[1][1]:6.0f}".ljust(10) + f"{avg_pre_cms[1][0]:6.0f}\n")
    file.write("".ljust(47) + f"{avg_pre_cms[0][1]:6.0f}".ljust(10) + f"{avg_pre_cms[0][0]:6.0f}\n")

    file.write("\n-------------------[SON MODEL POS REFINEMENT]------------------\n")
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

        file.write(f"Father models trained with: {config['fathers']['trained_at'].split('/')[-1]}\n")
        file.write(f"Father models module: {config['fathers']['module']}\n")
        file.write(f"Father models trained with: {config['fathers']['config']['training_mode']}\n")
        file.write(f"Son models module: {config['model']['module']}\n")
        file.write(f"Son models trained with: {config['model']['config']['training_mode']}\n")
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