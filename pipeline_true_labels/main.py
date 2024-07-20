import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from labeling_pipeline.dataset.data_loader import data_loader
from sklearn.metrics import confusion_matrix, accuracy_score
from cross_testing.dataset.data_leveler import data_leveler
from tools.utils.metrics import get_roc_auc, get_eer
from torchvision import transforms
from collections import Counter
import numpy as np
import importlib
import argparse
import torch
import math
import copy
import json

EARLY_STOP = 5

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def divide_dataset(dataset, days_qty):
    return (list(dataset.items())[:days_qty], list(dataset.items())[days_qty:])


def get_train_val_datasets(train_half, split, output_json):
    divider = math.floor(len(train_half) * split)

    train_ds, val_ds = train_half[:divider], train_half[divider:]
    train_ds, val_ds = [date[1] for date in train_ds], [date[1] for date in val_ds]
    train_ds, val_ds = torch.utils.data.ConcatDataset(train_ds), torch.utils.data.ConcatDataset(val_ds)

    total_train, total_val = len(train_ds), len(val_ds)

    before_class_0, before_class_1 = 0, 0
    for subset in train_ds.datasets:
        class_count = Counter(subset.targets)
        before_class_0 += class_count[0]
        before_class_1 += class_count[1]

    ds_leveler = data_leveler()
    flattened_train_ds = ds_leveler.flatten_dataset(train_ds)
    flattened_val_ds = ds_leveler.flatten_dataset(val_ds)

    after_class_0, after_class_1 = 0, 0
    for subset in flattened_train_ds.datasets:
        class_count = Counter(subset.targets)
        after_class_0 += class_count[0]
        after_class_1 += class_count[1]

    output_json["dataset"]["total_train"] = total_train
    output_json["dataset"]["total_val"] = total_val
    output_json["dataset"]["train_labels"] = {"before_leveling": {"empty": before_class_0, "occupied": before_class_1},
                                              "after_leveling": {"empty": after_class_0, "occupied": after_class_1}}

    return flattened_train_ds, flattened_val_ds


def get_test_dataset(test_half):
    test_ds = [date[1] for date in test_half]
    return torch.utils.data.ConcatDataset(test_ds)

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def test(model, test_ds, dl_config, threshold, output_json, output_model, output_type):
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)

    print(f"Testing model - {output_model} - {output_type}")
    preds, labels = model.predict(test_loader, threshold)
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    output_json[output_model][output_type] = {"accuracy": accuracy, "cm": str(cm)}


def get_threshold(model, val_ds, dl_config, output_json, output_model):
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)

    probs, labels = model.get_probs(val_loader)
    probs = np.array([prob[1] for prob in probs])

    roc, auc = get_roc_auc(probs, labels)
    threshold, eer = get_eer(roc)

    output_json[output_model]["metrics"] = {}
    output_json[output_model]["metrics"]["auc"] = float(auc)
    output_json[output_model]["metrics"]["eer"] = float(eer)
    output_json[output_model]["metrics"]["threshold"] = float(threshold)

    return threshold


def train(model, train_ds, val_ds, epochs, dl_config, output_json, output_name):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=dl_config["num_workers"], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)
    best_state_dict, best_loss, epoch_id, early_stop_count = None, math.inf, -1, 0

    print("Refining model")
    output_json["model"]["epochs"] = {}
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}] Initializing training epoch")

        model.train(train_loader)
        val_loss = model.get_loss_in_dataset(val_loader)

        if (best_loss - val_loss) > 0.0:
            best_state_dict = copy.deepcopy(model.get_state_dict())
            best_loss = val_loss
            epoch_id = (epoch + 1)
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        output_json["model"]["epochs"][f"epoch_{epoch + 1}"] = val_loss

        if early_stop_count == EARLY_STOP:
            print("5 attempts - val_loss not decreasing - early stoping")
            break
    
    output_json["model"]["best_model"] = {"loss": best_loss, "epoch_id": epoch_id, "model": f"{output_name}.pt"}

    return best_state_dict


def execute(model_module, config, args):
    model_name = args.initial.split('/')[-1][:-3]
    output_name = f"model_{args.subset}"
    output_json = {}

    selected_fields = ["img_size", "batch_size", "num_workers"]
    model_dl_config, model_config = config["model"]["dl_config"], config["model"]["config"]
    model_dl_config = { key: model_dl_config.get(key, config["experiment"].get(key)) for key in selected_fields }

    torch.cuda.set_device(model_config["device"])

    print(f"Starting Labeling Pipeline [INITIAL: {model_name}][SON: {config['model']['module']}][DATASET: {config['dataset']['path'].split('/')[-1]}][SUBSET: {args.subset}]")

    ds_loader = data_loader(model_dl_config)
    dataset = ds_loader.get_subset_from_dataset(config["dataset"]["path"], args.subset)
    train_half, test_half = divide_dataset(dataset, config["dataset"]["train_days"])

    output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
    output_json["dataset"] = {"dataset": config["dataset"]["path"].split('/')[-1], "subset": args.subset, "train_val_split": config["dataset"]["split"]}

    train_ds, val_ds = get_train_val_datasets(train_half, config["dataset"]["split"], output_json)
    test_ds = get_test_dataset(test_half)

    train_model = model_module.model(pre_trained=False, config=model_config)
    train_model.load_state_dict(torch.load(args.initial, map_location=model_config["device"]))

    output_json["model"] = {}
    output_json["model"]["initial_weights"] = args.initial
    output_json["model"]["details"] = model_config

    test(train_model, test_ds, model_dl_config, args.initial_threshold, output_json, "model", "pre_refinement")

    best_state_dict = train(train_model, train_ds, val_ds, config["experiment"]["epochs"], model_dl_config, output_json, output_name)
    test_model = model_module.model(pre_trained=False, config=model_config)
    test_model.load_state_dict(best_state_dict)

    test_threshold = get_threshold(test_model, val_ds, model_dl_config, output_json, "model")
    test(test_model, test_ds, model_dl_config, test_threshold, output_json, "model", "pos_refinement")

    print("Saving best model")
    torch.save(best_state_dict, f"_models/{args.exp}/{model_name}/{output_name}.pt")

    print("Saving result file")
    with open(f"_results/{args.exp}/{model_name}/{output_name}.json", "w") as output:
        json.dump(output_json, output, indent=2)


def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    model_module = importlib.import_module(config["model"]["module"])

    execute(model_module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--initial", "-i", type=str, required=True)
    parser.add_argument("--initial_threshold", "-it", type=float, required=True)
    parser.add_argument("--subset", "-su", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--exp", "-e", type=str, required=True)

    main(parser.parse_args())