import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder
import uuid

# Experiment variables
FATHERS_PATH = "/home/tteuz/Desktop/TCC/labeling_pipeline/fathers"
THRESHOLDS = [0.5846, 0.6202, 0.5366, 0.6524, 0.5149]
MODEL = "tools.models.mobilenet_v3"
LOSS = "CrossEntropyLoss"
DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/CNRParkEXTSegmented"
SUBSETS = ["CNR-CAMERA-1", "CNR-CAMERA-2", "CNR-CAMERA-3", "CNR-CAMERA-4", "CNR-CAMERA-5", "CNR-CAMERA-6", "CNR-CAMERA-7", "CNR-CAMERA-8", "CNR-CAMERA-9"]
EPOCHS = 15
SPLIT = 0.7

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"

fathers = [father for father in os.listdir(FATHERS_PATH) if ".pt" in father]

create_folder("_summaries")
create_folder("_models")
create_folder("_results")
create_folder(f"_models/{EXP_NAME}")
create_folder(f"_results/{EXP_NAME}")

# Writing experiment README
print("Writing experiment README")
with open(f"_results/exp_{EXP_UUID}/README.md", "w") as file:
    file.write("## Experiment infos\n")
    file.write("- Fathers models used:\n")
    for father in fathers:
        file.write(f"    - {father}\n")

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
for index, father in enumerate(fathers):
    create_folder(f"_models/{EXP_NAME}/{father[:-3]}")
    create_folder(f"_results/{EXP_NAME}/{father[:-3]}")
    for subset in SUBSETS:
        os.system(f"python3 main.py -f {FATHERS_PATH}/{father} -t {THRESHOLDS[index]} -m {MODEL} -l {LOSS} -d {DATASET} -su {subset} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")


# Writing summary
print("\nWriting summary")
os.system(f"python3 summary.py -f _results/{EXP_NAME}")
