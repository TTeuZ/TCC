import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder
import datetime
import uuid
import json

# Pipeline variables
FATHERS_PATH = "/home/tteuz/Desktop/TCC/labeling_pipeline/fathers/PKLot"
FATHERS_CONSTRUCT_JSON = json.load(open(f"{FATHERS_PATH}/construct.json"))

DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/CNRParkEXTSegmented"
SUBSETS = sorted(os.listdir(DATASET))

MODEL = "tools.models.mobilenet_v3"
LOSS = "CrossEntropyLoss"
EPOCHS = 15
SPLIT = 0.7

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"
EXP_DATETIME = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))


create_folder("_summaries")
create_folder("_models")
create_folder("_results")
create_folder(f"_models/{EXP_NAME}")
create_folder(f"_results/{EXP_NAME}")


# Writing experiment README
print("Writing experiment README")
with open(f"_results/exp_{EXP_UUID}/README.md", "w") as file:
    file.write(f"Date: {EXP_DATETIME}\n\n")
    file.write("## Experiment infos\n")
    file.write("- Fathers models used:\n")
    for father in FATHERS_CONSTRUCT_JSON["models"]:
        file.write(f"    - {father['name']}\n")

    file.write(f"- Model generated: {MODEL}\n")
    file.write(f"- Loss: {LOSS}\n")
    file.write(f"- Dataset: {DATASET}\n")

    file.write("- Subsets used:\n")
    for subset in SUBSETS:
        file.write(f"    - {subset}\n")
    
    file.write(f"- Training epocs: {EPOCHS}\n")
    file.write(f"- Train/Val split: {SPLIT}\n")
    file.write(f"- Sumary: _summaries/summary_{EXP_UUID}\n")


# Running pipeline
for father in FATHERS_CONSTRUCT_JSON["models"]:
    create_folder(f"_models/{EXP_NAME}/{father['name'][:-3]}")
    create_folder(f"_results/{EXP_NAME}/{father['name'][:-3]}")
    for subset in SUBSETS:
        os.system(f"python3 main.py -f {FATHERS_PATH}/{father['name']} -t {father['threshold']} -m {MODEL} -l {LOSS} -d {DATASET} -su {subset} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")


# Writing summary
print("\nWriting summary")
os.system(f"python3 summary.py -f _results/{EXP_NAME} -d {EXP_DATETIME} -m {MODEL} -md {FATHERS_CONSTRUCT_JSON['trained_at'].split('/')[-1]} -da {DATASET.split('/')[-1]}")