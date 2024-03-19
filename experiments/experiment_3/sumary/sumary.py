import numpy as np
import json
import os
import re

def get_cm(cm):
    temp = re.findall(r'\d+', cm)
    temp = np.array(temp, dtype=int).reshape(-1, 2)

    return temp


def print_results(results, file):
    results.sort(key=lambda result: result["test"]["average"]["loss"])

    best_result = results[0]
    best_avg = best_result["test"]["average"]
    best_subsets = best_result["test"]["subsets"]

    avg_losses = [result["test"]["average"]["loss"] for result in results]
    val_losses = [result["best_model"]["loss"] for result in results]
    epochs = [result["best_model"]["epoch_id"] for result in results]
    avg_accuracy = [result["test"]["average"]["accuracy"] for result in results]

    file.write("-------------------------[ALL RESULTS]-------------------------\n")
    file.write(" - ".join(f"{loss:.4f}" for loss in avg_losses) + " [LOSS]\n")
    file.write(" - ".join(f"{loss:.4f}" for loss in val_losses) + " [VAL LOSS]\n")
    file.write(" - ".join(f"{(epoch + 1):6d}" for epoch in epochs) + " [EPOCHS]\n")
    file.write(" - ".join(f"{acc:.4f}" for acc in avg_accuracy) + " [ACCURACY]\n")

    file.write("\n-------------------------[BEST RESULT]-------------------------\n")
    file.write(f"{best_avg['loss']:.4f} [LOSS]\n")
    file.write(f"{best_avg['accuracy']:.4f} [ACCURACY]\n")

    cm = get_cm(best_avg["cm"])

    file.write("\n")
    file.write("Confusion matrix:\n")
    for i in reversed(range(2)):
        for j in reversed(range(2)):
            file.write(f"{cm[i][j]} ")
        file.write("\n")

    file.write("\n-------------------------[ BY SUBSET ]-------------------------\n")
    subsets = [key for key in best_subsets.keys()]
    subsets.sort()
    for subset in subsets:
        cm = get_cm(best_subsets[subset]["cm"])

        file.write(f"{subset}: \t\t\t Confusion matrix\n")
        file.write(f"{best_subsets[subset]['loss']:.4f} [LOSS] \t\t\t {cm[1][1]} \t {cm[1][0]}\n")
        file.write(f"{best_subsets[subset]['accuracy']:.4f} [ACCURACY] \t\t {cm[0][1]} \t {cm[0][0]}\n")
        file.write("\n")


if __name__ == "__main__":
    results_path = "/home/tteuz/Desktop/TCC/experiments/experiment_3/_results"
    results = [(file, json.load(open(f"{results_path}/{file}", "r"))) for file in os.listdir(results_path)]
    splited_results = {"bcewithlogisticloss": {"exp_1": [], "exp_2": []}, "crossentropyloss": {"exp_1": [], "exp_2": []}, "bceloss": {"exp_1": [], "exp_2": []}}

    for result in results:
        loss = result[1]["model"]["loss"]
        exp = "exp_1" if result[1]["dataset"]["train"] == "PKLotSegmented" else "exp_2"

        splited_results[loss][exp].append(result[1])
    
    with open("bcewithlogisticloss.txt", "w") as file:
        results = splited_results["bcewithlogisticloss"]
        print_results(results["exp_1"], file)

        file.write("\n#############################################################################\n\n")
        print_results(results["exp_2"], file)

    with open("crossentropyloss.txt", "w") as file:
        results = splited_results["crossentropyloss"]
        print_results(results["exp_1"], file)

        file.write("\n#############################################################################\n\n")
        print_results(results["exp_2"], file)

    with open("bceloss.txt", "w") as file:
        results = splited_results["bceloss"]
        print_results(results["exp_1"], file)

        file.write("\n#############################################################################\n\n")
        print_results(results["exp_2"], file)