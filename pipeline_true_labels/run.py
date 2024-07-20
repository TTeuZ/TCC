import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder, get_related_models_json
import datetime
import argparse
import uuid
import json

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"

def main(args):
    assert os.path.exists(args.config), "Invalid config"

    begin_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    config = json.load(open(args.config, "r"))
    initial_weights = config["initial_weights"]
    subsets = sorted(os.listdir(config["dataset"]["path"]))

    initial_jsons = get_related_models_json(initial_weights["path"], initial_weights["dataset"])

    create_folder("_summaries")
    create_folder("_models")
    create_folder("_results")
    create_folder(f"_models/{EXP_NAME}")
    create_folder(f"_results/{EXP_NAME}")

    # Writing experiment README
    print("Writing experiment README")
    with open(f"_results/{EXP_NAME}/README.md", "w") as file:
        file.write("# Experiment infos\n\n")

        file.write("\n## Sons\n")
        file.write(f"- Model: {config['model']['module']}\n")
        file.write(f"- Loss: {config['model']['config']['loss']}\n")
        file.write(f"- Optimizer: {config['model']['config']['optimizer']}\n")

        file.write("\n## Experiment\n")
        file.write(f"- Dataset: {config['dataset']['path']}\n")
        file.write("- Subsets:\n")
        for subset in subsets:
            file.write(f"    - {subset}\n")
        
        file.write(f"- Training epocs: {config['experiment']['epochs']}\n")
        file.write(f"- Train/Val split: {config['dataset']['split']}\n")
        file.write(f"- Sumary: _summaries/summary_{EXP_UUID}\n")

    # Running pipeline
    for initial in initial_jsons:
        base_model = f"{initial_weights['path']}/models/{initial['best_model']['model']}"
        initial_threshold = initial["metrics"]["threshold"]

        create_folder(f"_models/{EXP_NAME}/{base_model.split('/')[-1][:-3]}")
        create_folder(f"_results/{EXP_NAME}/{base_model.split('/')[-1][:-3]}")

        for subset in subsets:
            os.system(f"python3 main.py -i {base_model} -it {initial_threshold} -su {subset} -c {args.config} -e {EXP_NAME}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    # Writing summary
    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{EXP_NAME} -bd {begin_datetime} -ed {end_datetime} -c {args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Testing Run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())