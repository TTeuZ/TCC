import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder
import argparse
import uuid
import json

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"

def main(args):
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    bases = sorted(os.listdir(config["models"]["weights"]["path"]))
    
    create_folder("_summaries")
    create_folder("_results")
    create_folder(f"_results/{EXP_NAME}")

    for base in bases:
        os.system(f"python3 main.py -b {config['models']['weights']['path']}/{base} -c {args.config} -e {EXP_NAME}")
    
    # Writing summary
    print("\nWriting summary")
    print(f"python3 summary.py -f _results/{EXP_NAME} -c {args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())