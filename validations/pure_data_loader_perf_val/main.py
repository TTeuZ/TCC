import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import confusion_matrix, accuracy_score
from tools.dataset.data_prefetcher import fast_collate
from tools.utils.metrics import get_roc_auc, get_eer
from data_leveler import data_leveler
from data_loader import data_loader
import numpy as np
import importlib
import argparse
import torch
import uuid
import json
import math
import copy

def test(model, test_ds, fc_config, threshold, output_json):
    collate_fn = lambda batch: fast_collate(batch, fc_config)

    output_json["test"] = { "subsets": {} }
    average_accuracy, final_cm = 0.0, [[0, 0], [0, 0]]

    print(f"Testing model")
    for test in test_ds:
        test_loader = torch.utils.data.DataLoader(test_ds[test], batch_size=fc_config["batch_size"], shuffle=False, num_workers=fc_config["num_workers"])
        preds, labels = model.predict(test_loader, threshold)

        accuracy = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        average_accuracy += accuracy
        final_cm += cm

        output_json["test"]["subsets"][test] = {"accuracy": accuracy, "cm": str(cm)}

    output_json["test"]["average"] = {"accuracy": (average_accuracy / len(test_ds)), "cm": str(final_cm)}


def get_threshold(model, val_ds, fc_config, output_json):
    collate_fn = lambda batch: fast_collate(batch, fc_config)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=fc_config["batch_size"], shuffle=False, num_workers=fc_config["num_workers"])

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
    collate_fn = lambda batch: fast_collate(batch, fc_config)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=fc_config["num_workers"])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=fc_config["batch_size"], shuffle=False, num_workers=fc_config["num_workers"])

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
    dl_config, fc_config, model_config = config
    output_name = f"model_{uuid.uuid4()}"

    output_json = {}

    print(f"Starting experiment [MODEL: {args.model}][TRAIN: {train_ds_name}][TEST: {test_ds_name}][TYPE: {args.type}]")

    ds_loader = data_loader(dl_config)
    train_ds, val_ds = ds_loader.load_dataset_by_split(args.train, args.split)
    test_ds = ds_loader.load_dataset_by_subset(args.test)

    ds_leveler = data_leveler()
    flattened_train_ds = ds_leveler.flatten_dataset(train_ds)
    flattened_val_ds = ds_leveler.flatten_dataset(val_ds)

    train_model = model_module.model(loss=args.loss, config=model_config)

    output_json["experiment"] = {"type": args.type, "dl_config": dl_config, "fc_config": fc_config, "model_config": model_config}
    output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
    output_json["dataset"] = {"train": train_ds_name, "test": test_ds_name, "train_val_split": args.split}
    output_json["model"] = train_model.info()
    output_json["model"]["loss"] = args.loss

    best_state_dict = train(train_model, flattened_train_ds, flattened_val_ds, args.epochs, fc_config, output_json, output_name)

    test_model = model_module.model(pre_trained=False, loss=args.loss, config=model_config)
    test_model.load_state_dict(best_state_dict)

    threshold = get_threshold(test_model, flattened_val_ds, fc_config, output_json)
    test(test_model, test_ds, fc_config, threshold, output_json)

    # print("Saving best model")
    # torch.save(best_state_dict, f"_models/{args.save}/{output_name}.pt")

    # print("Saving result file")
    # with open(f"_results/{args.save}/{output_name}.json", "w") as output:
    #     json.dump(output_json, output, indent=2)


def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.train), "Invalid train/val dataset"
    assert os.path.exists(args.test), "Invalid test dataset"
    assert args.type in ["fine_tunning", "transfer_learning"]
    assert 0 < args.split < 1, "Split may be between 0 and 1"
    assert args.epochs > 0, "Invalid epochs count"

    # Local Config
    if args.type == "fine_tunning":
        dl_config = { "img_size": (128, 128) }
        fc_config = { "img_size": (128, 128), "batch_size": 1000, "num_workers": 6 }
        model_config = { "training_mode": "normal", "normalize_data": True }
    else:
        dl_config = { "img_size": (224, 224) }
        fc_config = { "img_size": (224, 224), "batch_size": 400 }
        model_config = { "training_mode": "transfer", "normalize_data": True }
    
    # Server Config
    # if args.type == "fine_tunning":
    #     dl_config = { "img_size": (128, 128) }
    #     fc_config = { "img_size": (128, 128), "batch_size": 3500, "num_workers": 48 }
    #     model_config = { "training_mode": "normal", "normalize_data": True }
    # else:
    #     dl_config = { "img_size": (224, 224) }
    #     fc_config = { "img_size": (224, 224), "batch_size": 1400 }
    #     model_config = { "training_mode": "transfer", "normalize_data": True }
    
    config = (dl_config, fc_config, model_config)
    model_module = importlib.import_module(args.model)

    execute(model_module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--loss", "-l", type=str, required=True)
    parser.add_argument("--train", "-tr", type=str, required=True)
    parser.add_argument("--test", "-te", type=str, required=True)
    parser.add_argument("--split", "-s", type=float, required=True)
    parser.add_argument("--epochs", "-e", type=int, required=True)
    parser.add_argument("--save", "-sa", type=str, required=True)

    main(parser.parse_args())