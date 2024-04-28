import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils.utils import create_folder
import datetime
import uuid

# Experiment variables
# TRAIN_DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/PKLotSegmented" # Local path
# TEST_DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/CNRParkEXTSegmented" # Local path
TRAIN_DATASET = "/home/pmla20/datasets/PKLotSegmented" # Server path
TEST_DATASET = "/home/pmla20/datasets/CNRParkEXTSegmented" # Server path

MODEL = "tools.models.mobilenet_v3"
LOSS = "CrossEntropyLoss"
EPOCHS = 15
SPLIT = 0.7
TYPE = "fine_tunning"
# TYPE = "transfer_learning"

EXP_UUID = uuid.uuid4()
EXP_NAME = f"exp_{EXP_UUID}"
BEGIN_EXP_DATETIME = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

create_folder("_summaries")
create_folder("_models")
create_folder("_results")
create_folder(f"_models/{EXP_NAME}")
create_folder(f"_results/{EXP_NAME}")

# Writing experiment README
print("Writing experiment README")
with open(f"_results/{EXP_NAME}/README.md", "w") as file:
    file.write("## Experiment infos\n")
    file.write(f"- Cross testing type: {TYPE}\n")
    file.write(f"- Model: {MODEL}\n")
    file.write(f"- Train dataset: {TRAIN_DATASET}\n")
    file.write(f"- Test dataset: {TEST_DATASET}\n")
    file.write(f"- Training epochs: {EPOCHS}\n")
    file.write(f"- Train/Val split: {SPLIT}\n")
    file.write(f"- Summary: _summaries/summary_{EXP_UUID}\n")

# Running experiments (Cross testing -> 5 times training with TRAIN_DATASET and testing with TEST_DATASET and 5 time inverting datasets)
print("\nExecuting experiment")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")

os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")
os.system(f"python3 main.py -t {TYPE} -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {EXP_NAME}")

END_EXP_DATETIME = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

# Writing summary
print("\nWriting summary")
os.system(f"python3 summary.py -f _results/{EXP_NAME} -bd {BEGIN_EXP_DATETIME} -ed {END_EXP_DATETIME} -t {TYPE} -m {MODEL} -l {LOSS} -s {SPLIT}")