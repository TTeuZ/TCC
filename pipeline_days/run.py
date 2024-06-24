import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from labeling_pipeline.models.ensemble_generator import ensemble_generator
from tools.utils.utils import create_folder
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
    fathers_json = config['fathers']
    initial_weights = config["initial_weights"]
    subsets = sorted(os.listdir(config["dataset"]["path"]))

    generator = ensemble_generator(config, EXP_NAME)

    create_folder("_summaries")
    create_folder("_results")
    create_folder(f"_results/{EXP_NAME}")

    # Writing experiment README
    print("Writing experiment README")
    with open(f"_results/{EXP_NAME}/README.md", "w") as file:
        file.write("# Experiment infos\n\n")

        file.write("## Fathers\n")
        file.write(f"- Model: {fathers_json['module']}\n")
        file.write(f"- trained_At: {fathers_json['trained_at']}\n")
        file.write(f"- Training mode: {fathers_json['config']['training_mode']}\n")
        file.write("- Models used:\n")
        for father in fathers_json["models"]:
            file.write(f"    - {father}\n")

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
        file.write(f"- Train days: {' - '.join(str(day) for day in config['dataset']['train_days'])}\n")
        file.write(f"- val days: {config['dataset']['val_days']}\n")
        file.write(f"- Sumary: _summaries/summary_{EXP_UUID}\n")

    # Running pipeline
    for index, father in enumerate(fathers_json["models"]):
        create_folder(f"_results/{EXP_NAME}/{father.split('/')[-1][:-3]}")

        generator.generate(father)
        for subset in subsets:
            os.system(f"python3 main.py -f {father} -i {initial_weights[index]['name']} -it {initial_weights[index]['threshold']} -su {subset} -c {args.config} -e {EXP_NAME}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    # Writing summary
    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{EXP_NAME} -bd {begin_datetime} -ed {end_datetime} -c {args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Testing Run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())