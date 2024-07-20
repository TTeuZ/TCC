import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from collections import defaultdict
# from datetime import datetime
# import numpy as np
import argparse
# import json

# def print_results(subset, results, file):
#     all_images = results[0]["dataset"]["classify"]["all_images"]
#     used_images = [result["dataset"]["classify"]["used_images"] for result in results]
#     wrong_labeled = [result["dataset"]["classify"]["wrong_labels"] for result in results]
#     empty_wrong = [result["dataset"]["classify"]["empty_wrong"] for result in results]
#     occupied_wrong = [result["dataset"]["classify"]["occupied_wrong"] for result in results]

#     avg_used_images = int(np.round(sum(used_images) / len(used_images)))
#     avg_wrong_labeled = int(np.round(sum(wrong_labeled) / len(wrong_labeled)))
#     avg_empty_wrong = int(sum(empty_wrong) / len(empty_wrong))
#     avg_occupied_wrong = int(sum(occupied_wrong) / len(occupied_wrong))

#     std_used_images = np.std(used_images)
#     pct_used_images = (avg_used_images * 100) / all_images

#     std_wrong_labeled = np.std(wrong_labeled)
#     pct_wrong_labeled = (avg_wrong_labeled * 100) / avg_used_images

#     file.write(f"Subset: {subset.split('_')[1]}\n")
#     file.write("\n---------------------[IMAGES FOR TRAINING]---------------------\n")
#     file.write(f"All: {all_images} - Used: {avg_used_images}[AVG] - {std_used_images:.2f}[STD] - {pct_used_images:.2f}[%]\n")
#     file.write(f"Wrong Labeled: {avg_wrong_labeled}[AVG] - {std_wrong_labeled:.2f}[STD] - {pct_wrong_labeled:.2f}[%]\n")
#     file.write(f"Empty wrong: {avg_empty_wrong}[AVG] - Occupied wrong: {avg_occupied_wrong}[AVG]\n\n")

#     file.write("\n------------------------[FATHER MODEL]-------------------------\n")
#     father_accs = [result["father_model"]["test"]["accuracy"] for result in results]

#     father_avg_acc = sum(father_accs) / len(father_accs)
#     father_std_acc = np.std(father_accs)

#     file.write(" [AVG]  [STD]\n")
#     file.write(f"{father_avg_acc:.4f} {father_std_acc:.4f} [ACCURACY]\n")

#     file.write("\n-------------------[SON MODEL PRE REFINEMENT]------------------\n")
#     pre_accs = [result["model"]["pre_refinement"]["accuracy"] for result in results]
#     avg_pre_acc = sum(pre_accs) / len(pre_accs)
#     std_pre_acc = np.std(pre_accs)

#     file.write(" [AVG]  [STD]\n")
#     file.write(f"{avg_pre_acc:.4f} {std_pre_acc:.4f} [ACCURACY]\n")

#     file.write("\n-------------------[SON MODEL POS REFINEMENT]------------------\n")
#     days = [value for value in results[0]["model"].keys() if "days" in value]
    
#     file.write(" [AVG]  [STD]\n")
#     for day in days:
#         pos_accs = [result["model"][day]["pos_refinement"]["accuracy"] for result in results]
#         avg_pos_acc = sum(pos_accs) / len(pos_accs)
#         std_pos_acc = np.std(pos_accs)

#         file.write(f"{avg_pos_acc:.4f} {std_pos_acc:.4f} [ACCURACY {day.upper()}]\n")

#     file.write(f"\n")


# def print_avg_results(results_by_subset, file):
#     all_images, used_images = 0, defaultdict(list)
#     wrong_labeled, empty_wrong, occupied_wrong = defaultdict(list), defaultdict(list), defaultdict(list)
#     father_accs, pre_accs, pos_accs = [], [], defaultdict(list)

#     for subset, results in results_by_subset.items():
#         thresholds = [value for value in results[0]["model"].keys() if "classify" in value]
        
#         all_images += results_by_subset[subset][0]["model"][thresholds[0]]["all_images"]
#         subset_used_images = defaultdict(list)

#         for result in results:
#             father_accs.append(result["father_model"]["test"]["accuracy"])
#             pre_accs.append(result["model"]["pre_refinement"]["accuracy"])

#             for threshold in thresholds:
#                 subset_used_images[threshold].append(result["model"][threshold]["used_images"])
#                 wrong_labeled[threshold].append(result["model"][threshold]["wrong_labels"])
#                 empty_wrong[threshold].append(result["model"][threshold]["empty_wrong"])
#                 occupied_wrong[threshold].append(result["model"][threshold]["occupied_wrong"])
#                 pos_accs[threshold].append(result["model"][threshold]["pos_refinement"]["accuracy"])

#         for threshold in thresholds:
#             subset_used_images[threshold] = int(np.round(sum(subset_used_images[threshold]) / len(subset_used_images[threshold])))
#             used_images[threshold].append(subset_used_images[threshold])


# def get_results_by_subset(path):
#     models_results = [folder for folder in os.listdir(path) if "model" in folder]

#     results_by_subset = defaultdict(list)
#     for model in models_results:
#         jsons = os.listdir(f"{path}/{model}")
#         results = [json.load(open(f"{path}/{model}/{file}", "r")) for file in jsons]
#         for result in results:
#             results_by_subset[f"model_{result['dataset']['subset']}"].append(result)

#     results_by_subset = dict(results_by_subset)
#     return dict(sorted(results_by_subset.items()))


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"
    assert os.path.exists(args.config), "Invalid config"

    print("TBD")

    # config = json.load(open(args.config, "r"))
    # results_by_subset = get_results_by_subset(args.files)

    # begin_date = datetime.strptime(args.begin_date, '%Y-%m-%d--%H:%M:%S')
    # end_date = datetime.strptime(args.end_date, '%Y-%m-%d--%H:%M:%S')
    # time_spent = end_date - begin_date

    # summary_name = args.files.split("/")[1]
    # with open(f"_summaries/{summary_name}.txt", "w") as file:
    #     file.write(f"Begin: {args.begin_date} End: {args.end_date} -- Time spent: {time_spent} \n\n")

    #     file.write(f"Details in {args.files}")
    #     file.write("\n---------------------------------------------------------------\n\n")

    #     file.write(f"Father models trained with: {config['fathers']['trained_at']}\n")
    #     file.write(f"Father models module: {config['fathers']['module']}\n")
    #     file.write(f"Father models trained with: {config['fathers']['config']['training_mode']}\n")
    #     file.write(f"Son models module: {config['model']['module']}\n")
    #     file.write(f"Son models trained with: {config['model']['config']['training_mode']}\n")
    #     file.write(f"Dataset used: {config['dataset']['path']}\n")
    #     file.write(f"Training days: {config['dataset']['train_days']}\n")
    #     file.write(f"Split: {config['dataset']['split']}\n")

    #     file.write("\n###############################################################\n\n")

    #     print_avg_results(results_by_subset, file)

    #     file.write("\n###############################################################\n\n")
    #     for subset, results in results_by_subset.items():
    #         print_results(subset, results, file)
    #         file.write("###############################################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    
    main(parser.parse_args())