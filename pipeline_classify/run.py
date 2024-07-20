import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from labeling_pipeline.models.ensemble_generator import ensemble_generator
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
    fathers_config = config['fathers']
    initial_weights = config["initial_weights"]
    subsets = sorted(os.listdir(config["dataset"]["path"]))

    father_jsons = get_related_models_json(fathers_config["models"]["path"], fathers_config["models"]["dataset"])
    initial_jsons = get_related_models_json(initial_weights["path"], initial_weights["dataset"])
    generator = ensemble_generator(config, EXP_NAME)

    create_folder("_summaries")
    create_folder("_results")
    create_folder(f"_results/{EXP_NAME}")

    # Running pipeline
    for index, father in enumerate(father_jsons):
        father_model = f"{fathers_config['models']['path']}/models/{father['best_model']['model']}"
        create_folder(f"_results/{EXP_NAME}/{father_model.split('/')[-1][:-3]}")

        generator.generate(father_model)
        for subset in subsets:
            base_model = f"{initial_weights['path']}/models/{initial_jsons[index]['best_model']['model']}"
            initial_threshold = initial_jsons[index]["metrics"]["threshold"]
            os.system(f"python3 main.py -f {father_model} -i {base_model} -it {initial_threshold} -su {subset} -c {args.config} -e {EXP_NAME}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    # Writing summary
    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{EXP_NAME} -bd {begin_datetime} -ed {end_datetime} -c {args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Testing Run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())