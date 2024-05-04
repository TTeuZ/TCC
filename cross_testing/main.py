import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import confusion_matrix, accuracy_score
from tools.utils.metrics import get_roc_auc, get_eer
from dataset.data_leveler import data_leveler
from dataset.data_loader import data_loader
import numpy as np
import importlib
import argparse
import torch
import uuid
import json
import math
import copy

def test(model, test_ds, fc_config, threshold, output_json):
    average_accuracy, final_cm = 0.0, [[0, 0], [0, 0]]
    output_json["test"] = { "subsets": {} }

    print(f"Testing model")
    for test in test_ds:
        test_loader = torch.utils.data.DataLoader(test_ds[test], batch_size=fc_config["batch_size"], shuffle=False, num_workers=fc_config["num_workers"], pin_memory=True)
        preds, labels = model.predict(test_loader, threshold)

        accuracy = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        average_accuracy += accuracy
        final_cm += cm

        output_json["test"]["subsets"][test] = {"accuracy": accuracy, "cm": str(cm)}

    output_json["test"]["average"] = {"accuracy": (average_accuracy / len(test_ds)), "cm": str(final_cm)}


def get_threshold(model, val_ds, fc_config, output_json):
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=fc_config["batch_size"], shuffle=False, num_workers=fc_config["num_workers"], pin_memory=True)

    probs, labels = model.get_probs(val_loader)
    probs = np.array([prob[1] for prob in probs])

    roc, auc = get_roc_auc(probs, labels)
    threshold, eer = get_eer(roc)

    output_json["metrics"] = {}
    output_json["metrics"]["auc"] = float(auc)
    output_json["metrics"]["eer"] = float(eer)
    output_json["metrics"]["threshold"] = float(threshold)

    return threshold


def train(model, train_ds, val_ds, epochs, fc_config, output_json, output_name):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=fc_config["num_workers"], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=fc_config["batch_size"], shuffle=False, num_workers=fc_config["num_workers"], pin_memory=True)

    best_state_dict, best_loss, epoch_id = None, math.inf, -1

    output_json["epochs"] = {}
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}] Initializing training epoch")

        model.train(train_loader)
        val_loss = model.get_loss_in_dataset(val_loader)

        if (best_loss - val_loss) > 0.0:
            best_state_dict = copy.deepcopy(model.get_state_dict())
            best_loss = val_loss
            epoch_id = (epoch + 1)
        
        output_json["epochs"][f"epoch_{epoch + 1}"] = val_loss
    
    output_json["best_model"] = {"loss": best_loss, "epoch_id": epoch_id, "model": f"{output_name}.pt"}

    return best_state_dict


def execute(model_module, config, args):
    train_ds_name, test_ds_name  = args.train.split("/")[-1], args.test.split("/")[-1]
    output_name = f"model_{uuid.uuid4()}"
    output_json = {}

    dl_config = config["experiment"]["dl_config"]
    fc_config = config["experiment"]["fc_config"]
    model_config = config["model"]["config"]

    torch.cuda.set_device(model_config["device"])

    print(f"Starting experiment [MODEL: {config['model']['module']}][TRAIN: {train_ds_name}][TEST: {test_ds_name}][TYPE: {config['experiment']['type']}]")

    ds_loader = data_loader(dl_config)
    train_ds, val_ds = ds_loader.load_dataset_by_split(args.train, config["datasets"]["split"])
    test_ds = ds_loader.load_dataset_by_subset(args.test)

    ds_leveler = data_leveler()
    flattened_train_ds = ds_leveler.flatten_dataset(train_ds)
    flattened_val_ds = ds_leveler.flatten_dataset(val_ds)

    train_model = model_module.model(config=model_config)

    output_json["experiment"] = {"type": config["experiment"]["type"], "dl_config": dl_config, "fc_config": fc_config, "model_config": model_config}
    output_json["experiment"]["model_config"]["module"] = config["model"]["module"]
    output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
    output_json["dataset"] = {"train": train_ds_name, "test": test_ds_name, "train_val_split": config["datasets"]["split"]}

    best_state_dict = train(train_model, flattened_train_ds, flattened_val_ds, config["experiment"]["epochs"], fc_config, output_json, output_name)

    test_model = model_module.model(pre_trained=False, config=model_config)
    test_model.load_state_dict(best_state_dict)

    threshold = get_threshold(test_model, flattened_val_ds, fc_config, output_json)
    test(test_model, test_ds, fc_config, threshold, output_json)

    print("Saving best model")
    torch.save(best_state_dict, f"_models/{args.save}/{output_name}.pt")

    print("Saving result file")
    with open(f"_results/{args.save}/{output_name}.json", "w") as output:
        json.dump(output_json, output, indent=2)


def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    model_module = importlib.import_module(config["model"]["module"])

    execute(model_module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", "-tr", type=str, required=True)
    parser.add_argument("--test", "-te", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    main(parser.parse_args())