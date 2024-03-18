from utils.utils import create_folder
import uuid
import os

# Experiment variables
MODEL = "models.mobilenet_v3"
LOSS = "CrossEntropyLoss"
TRAIN_DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/PKLotSegmented"
TEST_DATASET = "/media/tteuz/ssd/datasets/PKLot2.0/CNRParkEXTSegmented"
EPOCHS = 15
SPLIT = 0.7

EXP_UUID = uuid.uuid4()
SAVE_LOCATION = f"exp_{EXP_UUID}"


create_folder("_summaries")
create_folder("_models")
create_folder("_results")
create_folder(f"_models/{SAVE_LOCATION }")
create_folder(f"_results/{SAVE_LOCATION }")


# Writing experiment README
print("Writing experiment README")
with open(f"_results/exp_{EXP_UUID}/README.md", "w") as file:
    file.write("## Experiment infos\n")
    file.write(f"Model: {MODEL}\n")
    file.write(f"Train dataset: {TRAIN_DATASET}\n")
    file.write(f"Test dataset: {TEST_DATASET}\n")
    file.write(f"Training epocs: {EPOCHS}\n")
    file.write(f"Train/Val split: {SPLIT}\n")
    file.write(f"Sumary: _summaries/summary_{EXP_UUID}\n")


# Running experiments (Cross testing -> 5 times training with TRAIN_DATASET and testing with TEST_DATASET and 5 time inverting datasets)
print("\nExecuting experiment")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TRAIN_DATASET} -te {TEST_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")

os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")
os.system(f"python3 main.py -m {MODEL} -l {LOSS} -tr {TEST_DATASET} -te {TRAIN_DATASET} -s {SPLIT} -e {EPOCHS} -sa {SAVE_LOCATION}")


# Writing summary
print("\nWriting summary")
os.system(f"python3 summary.py -f _results/{SAVE_LOCATION} -m {MODEL} -l {LOSS} -s {SPLIT}")