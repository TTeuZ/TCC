import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder
import uuid

# Experiment variables
FATHERS_PATH = "/home/tteuz/Desktop/TCC/labeling_pipeline/fathers"
FATHERS = ["model_8e9989a8-1589-44b2-8f11-41a1bbc47234.pt", "model_47fc46ce-4bfd-43d9-bc3a-30ebcd67633a.pt", 
           "model_000433a1-27bd-4532-8241-354d89b5c854.pt", "model_4388d475-d38f-455b-926f-f6cc82b894fe.pt", 
           "model_c9ca7683-15a6-43d8-be2f-3e8874c08475.pt"]
THRESHOLDS = [0.652393639087677, 0.514877200126648, 0.5845631957054138, 0.5365752577781677, 0.6202115416526794]

MODEL = "tools.models.mobilenet_v3"
LOSS = "CrossEntropyLoss"
DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/CNRParkEXTSegmented"
SUBSETS = ["CNR-CAMERA-1", "CNR-CAMERA-2", "CNR-CAMERA-3", "CNR-CAMERA-4", "CNR-CAMERA-5", "CNR-CAMERA-6", "CNR-CAMERA-7", "CNR-CAMERA-8", "CNR-CAMERA-9"]
EPOCHS = 15
SPLIT = 0.7

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"

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
    for father in FATHERS:
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
for index, father in enumerate(FATHERS):
    create_folder(f"_models/{EXP_NAME}/{father[:-3]}")
    create_folder(f"_results/{EXP_NAME}/{father[:-3]}")
    for subset in SUBSETS:
        os.system(f"python3 main.py -f {FATHERS_PATH}/{father} -t {THRESHOLDS[index]} -m {MODEL} -l {LOSS} -d {DATASET} -su {subset} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")


# Writing summary
print("\nWriting summary")
os.system(f"python3 summary.py -f _results/{EXP_NAME}")
