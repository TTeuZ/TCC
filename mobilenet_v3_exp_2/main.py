from model.mobilenet_v3 import mobilenet_v3
from dataset.data_loader import data_loader
import model.utils as utils
import argparse
import torch
import json
import math
import copy
import os

# Fixed num epochs for training for now
NUM_EPOCHS = 10

def training(model, train_ds, val_ds, output_json, output_name, index):
    collate_fn = lambda batch: utils.fast_collate(batch)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=6, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn, pin_memory=True)
    
    best_state_dict, best_loss, epoch_id = None, math.inf, -1

    output_json["epochs"] = []
    for epoch in range(NUM_EPOCHS):
        print(f"[REPEAT {index + 1}] Epoch [{epoch + 1}/{NUM_EPOCHS}] Initializing training epoch")
        epoch_output = {}

        train_loss = model.train(train_loader)
        val_loss, accuracy, cm = model.predict(val_loader)

        if (best_loss - val_loss) > 0.0:
            best_state_dict = copy.deepcopy(model.get_state_dict())
            best_loss = val_loss
            epoch_id = epoch
        
        epoch_output["train_loss"] = train_loss
        epoch_output["val_loss"] = val_loss
        epoch_output["val_accuracy"] = accuracy
        epoch_output["val_cm"] = str(cm)
        output_json["epochs"].append(epoch_output)
        
    output_json["best_model"] = {"loss": best_loss, "epoch_id": epoch_id, "model_path": f"_models/{output_name}.pt"}
    
    print(f"[REPEAT {index + 1}] Saving best model")
    torch.save(model.get_state_dict(), f"_models/{output_name}.pt")

    return best_state_dict


def testing(model, test_ds, output_json, index):
    collate_fn = lambda batch: utils.fast_collate(batch)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn, pin_memory=True)

    print(f"[REPEAT {index + 1}] Testing model")
    loss, accuracy, cm = model.predict(test_loader)

    output_json["test"] = {"loss": loss, "accuracy": accuracy, "cm": str(cm)}


def execute(args):
    train_dataset_name = args.train.split("/")[-1]
    test_dataset_name = args.test.split("/")[-1]

    ds_loader = data_loader()
    train_ds, val_ds = ds_loader.load_dataset(args.train, args.split)
    test_ds = ds_loader.load_dataset(args.test)

    for index in range(args.repeat):
        print(f"[REPEAT {index + 1}] Initializing process")
        output_name = f"train_{train_dataset_name}_test_{test_dataset_name}_{index}"
        output_json = {}
        output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
        output_json["dataset"] = {"train": train_dataset_name, "test": test_dataset_name, "train_val_split": args.split}

        model = mobilenet_v3()
        best_state_dict = training(model, train_ds, val_ds, output_json, output_name, index)

        test_model = mobilenet_v3(pre_trained=False)
        test_model.load_state_dict(best_state_dict)

        testing(test_model, test_ds, output_json, index)

        with open(f"_results/{output_name}.json", "w") as output:
            json.dump(output_json, output, indent=2)


def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.train), "Invalid train/val dataset"
    assert os.path.exists(args.test), "Invalid test dataset"
    assert 0 < args.split < 1, "Split may be between 0 and 1"
    assert args.repeat > 0, "Invalid repeat count"
    
    if not os.path.exists("_models"):
        os.mkdir("_models")

    if not os.path.exists("_results"):
        os.mkdir("_results")
    
    execute(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", "-tr", type=str, required=True)
    parser.add_argument("--test", "-te", type=str, required=True)
    parser.add_argument("--split", "-s", type=float, required=True)
    parser.add_argument("--repeat", "-r", type=int, required=True)

    main(parser.parse_args())

# Train with PKLot2.0 and test with CNRPark-EXT 5 times with 0.7 train size
# python3 main.py -tr /home/tteuz/Desktop/TCC/datasets/PKLot2.0/PKLotSegmented -te /home/tteuz/Desktop/TCC/datasets/PKLot2.0/CNRPartEXTSegmented -s 0.7 -r 5
    
# Train with CNRPark-EXT and test with PKLot 5 times with 0.7 train size
# python3 main.py -tr /home/tteuz/Desktop/TCC/datasets/PKLot2.0/CNRPartEXTSegmented -te /home/tteuz/Desktop/TCC/datasets/PKLot2.0/PKLotSegmented -s 0.7 -r 5