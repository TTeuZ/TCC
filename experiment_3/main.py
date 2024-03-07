from dataset.data_loader import data_loader
from utils import create_folder
import numpy as np
import importlib
import argparse
import torch
import json
import math
import copy
import uuid
import os

def fast_collate(batch):
    images = [image[0] for image in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.float32)

    width, height = 128, 128
    tensor = torch.zeros((len(images), 3, height, width), dtype=torch.float32).contiguous()

    for index, image in enumerate(images):
        nump_array = np.asarray(image, dtype=np.float32)

        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2)
        tensor[index] += torch.from_numpy(nump_array)

    return tensor, targets


def train(model, train_ds, val_ds, output_json, output_name, num_epochs):
    collate_fn = lambda batch: fast_collate(batch)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=6, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn)
    
    best_state_dict, best_loss, epoch_id = None, math.inf, -1

    output_json["epochs"] = []
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] Initializing training epoch")
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

    return best_state_dict


def test(model, test_ds, output_json):
    collate_fn = lambda batch: fast_collate(batch)

    output_json["test"] = { "subsets": {} }
    average_loss, average_accuracy, final_cm = 0.0, 0.0, [[0, 0], [0, 0]]

    print(f"Testing model")
    for test in test_ds:
        test_loader = torch.utils.data.DataLoader(test_ds[test], batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn)

        loss, accuracy, cm = model.predict(test_loader)

        average_loss += loss
        average_accuracy += accuracy
        final_cm += cm

        output_json["test"]["subsets"][test] = {"loss": loss, "accuracy": accuracy, "cm": str(cm)}

    output_json["test"]["average"] = {"loss": (average_loss / len(test_ds)), "accuracy": (average_accuracy / len(test_ds)), "cm": str(final_cm)}


def execute(model_module, args):
    train_dataset_name = args.train.split("/")[-1]
    test_dataset_name = args.test.split("/")[-1]
    output_json = {}

    print(f"Starting experiment [MODEL: {args.model}][TRAIN: {train_dataset_name}][TEST: {test_dataset_name}]")

    output_name = f"model_{uuid.uuid4()}"
    output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
    output_json["dataset"] = {"train": train_dataset_name, "test": test_dataset_name, "train_val_split": args.split}

    ds_loader = data_loader()
    model = model_module.mobilenet_v3()

    output_json["model"] = model.info()
    output_json["model"]["loss"] = args.model.split("_")[1]

    train_ds, val_ds = ds_loader.load_dataset_as_train(args.train, args.split)
    best_state_dict = train(model, train_ds, val_ds, output_json, output_name, args.epochs)

    test_model = model_module.mobilenet_v3(pre_trained=False)
    test_model.load_state_dict(best_state_dict)

    test_ds = ds_loader.load_dataset_as_test(args.test)
    test(test_model, test_ds, output_json)

    print("Saving best model")
    torch.save(best_state_dict, f"_models/{output_name}.pt")

    print("Saving result file")
    with open(f"_results/{output_name}.json", "w") as output:
        json.dump(output_json, output, indent=2)
        

def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.train), "Invalid train/val dataset"
    assert os.path.exists(args.test), "Invalid test dataset"
    assert 0 < args.split < 1, "Split may be between 0 and 1"
    assert args.epochs > 0, "Invalid epochs count"
    model_module = importlib.import_module(args.model)

    create_folder("_models")
    create_folder("_results")
    
    execute(model_module, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--train", "-tr", type=str, required=True)
    parser.add_argument("--test", "-te", type=str, required=True)
    parser.add_argument("--split", "-s", type=float, required=True)
    parser.add_argument("--epochs", "-e", type=int, required=True)

    main(parser.parse_args())