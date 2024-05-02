import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder
import datetime
import argparse
import uuid
import json

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"

def main(args):
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    begin_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    create_folder("_summaries")
    create_folder("_models")
    create_folder("_results")
    create_folder(f"_models/{EXP_NAME}")
    create_folder(f"_results/{EXP_NAME}")

    print("Writing experiment README")
    with open(f"_results/{EXP_NAME}/README.md", "w") as file:
        file.write("## Experiment infos\n")
        file.write(f"- Cross testing type: {config['experiment']['type']}\n")
        file.write(f"- Model: {config['model']['module']}\n")
        file.write(f"- Train dataset: {config['datasets']['train_dataset']}\n")
        file.write(f"- Test dataset: {config['datasets']['test_dataset']}\n")
        file.write(f"- Training epochs: {config['experiment']['epochs']}\n")
        file.write(f"- Train/Val split: {config['datasets']['split']}\n")
        file.write(f"- Summary: _summaries/summary_{EXP_UUID}\n")

    print("\nExecuting experiment")
    os.system(f"python3 main.py -tr {config['datasets']['train_dataset']} -te {config['datasets']['test_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['train_dataset']} -te {config['datasets']['test_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['train_dataset']} -te {config['datasets']['test_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['train_dataset']} -te {config['datasets']['test_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['train_dataset']} -te {config['datasets']['test_dataset']} -c {args.config} -s {EXP_NAME}")

    os.system(f"python3 main.py -tr {config['datasets']['test_dataset']} -te {config['datasets']['train_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['test_dataset']} -te {config['datasets']['train_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['test_dataset']} -te {config['datasets']['train_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['test_dataset']} -te {config['datasets']['train_dataset']} -c {args.config} -s {EXP_NAME}")
    os.system(f"python3 main.py -tr {config['datasets']['test_dataset']} -te {config['datasets']['train_dataset']} -c {args.config} -s {EXP_NAME}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{EXP_NAME} -bd {begin_datetime} -ed {end_datetime} -t {config['experiment']['type']} -m {config['model']['module']} -l {config['model']['config']['loss']} -o {config['model']['config']['optimizer']} -s {config['datasets']['split']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Testing Run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())