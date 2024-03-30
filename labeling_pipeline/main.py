import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import confusion_matrix, accuracy_score
from tools.dataset.data_prefetcher import fast_collate
from tools.utils.metrics import get_roc_auc, get_eer
from dataset.data_loader import data_loader
import numpy as np
import importlib
import argparse
import torch
import math
import copy
import json

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def divide_dataset(dataset, split=0.5):
    divisor = math.ceil(len(dataset) * split)
    return (list(dataset.items())[:divisor], list(dataset.items())[divisor:])


def get_train_val_datasets(ds_loader, first_half, new_labels, split=0.7):
    divisor = math.ceil(len(first_half) * split)

    train_ds, train_labels = first_half[:divisor], new_labels[:divisor]
    val_ds, val_labels = first_half[divisor:], new_labels[divisor:]

    train_ds, train_labels = [date[1] for date in train_ds], [label[1] for label in train_labels]
    val_ds, val_labels = [date[1] for date in val_ds], [label[1] for label in val_labels]

    train = ds_loader.concat_dataset_with_new_labels(train_ds, train_labels)
    val = ds_loader.concat_dataset_with_new_labels(val_ds, val_labels)

    return train, val


def get_test_dataset(ds_loader, second_half):
    test_ds = [date[1] for date in second_half]
    return ds_loader.concat_dataset(test_ds)

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def test(model, test_ds, threshold, output_json, output_model):
    collate_fn = lambda batch: fast_collate(batch)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn)

    preds, labels = model.predict(test_loader, threshold)
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    output_json[output_model]["test"] = {"accuracy": accuracy, "cm": str(cm)}


def get_threshold(model, val_ds, output_json):
    collate_fn = lambda batch: fast_collate(batch)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn)

    probs, labels = model.get_probs(val_loader)
    probs = np.array([prob[1] for prob in probs])

    roc, auc = get_roc_auc(probs, labels)
    threshold, eer = get_eer(roc)

    output_json["model"]["metrics"] = {}
    output_json["model"]["metrics"]["auc"] = float(auc)
    output_json["model"]["metrics"]["eer"] = float(eer)
    output_json["model"]["metrics"]["threshold"] = float(threshold)

    return threshold


def train(model, train_ds, val_ds, output_json, output_name, epochs):
    collate_fn = lambda batch: fast_collate(batch)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=6, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn)

    print("Training model")

    best_state_dict, best_loss, epoch_id = None, math.inf, -1

    output_json["model"]["epochs"] = {}
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}] Initializing training epoch")

        model.train(train_loader)
        val_loss = model.get_loss_in_dataset(val_loader)

        if (best_loss - val_loss) > 0.0:
            best_state_dict = copy.deepcopy(model.get_state_dict())
            best_loss = val_loss
            epoch_id = (epoch + 1)
        
        output_json["model"]["epochs"][f"epoch_{epoch + 1}"] = val_loss
    
    output_json["model"]["best_model"] = {"loss": best_loss, "epoch_id": epoch_id, "model": f"{output_name}.pt"}

    return best_state_dict


def classify(father_model, threshold, first_half):
    collate_fn = lambda batch: fast_collate(batch)

    print("Classifing images")

    all_preds = []
    for date in first_half:
        classify_loader = torch.utils.data.DataLoader(date[1], batch_size=1000, shuffle=False, num_workers=6, collate_fn=collate_fn)
        preds, _ = father_model.predict(classify_loader, threshold)
        all_preds.append((date[0], preds))
    
    return all_preds


def execute(model_module, args):
    father_model_name = args.father.split('/')[-1][:-3]
    output_name = f"model_{args.subset}"
    output_json = {}

    print(f"Starting Labeling Pipeline [FATHER: {father_model_name}][DATASET: {args.dataset.split('/')[-1]}][SUBSET: {args.subset}]")

    ds_loader = data_loader()
    dataset = ds_loader.get_subset_from_dataset(args.dataset, args.subset)
    first_half, second_half = divide_dataset(dataset)

    father_model = model_module.model(pre_trained=False, loss=args.loss)
    father_model.load_state_dict(torch.load(args.father))

    output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
    output_json["dataset"] = {"dataset": args.dataset.split('/')[-1], "subset": args.subset, "train_val_split": args.split}
    output_json["father_model"] = father_model.info()
    output_json["father_model"]["name"] = father_model_name
    output_json["father_model"]["threshold"] = args.threshold

    new_labels = classify(father_model, args.threshold, first_half)
    train_ds, val_ds = get_train_val_datasets(ds_loader, first_half, new_labels, args.split)
    test_ds = get_test_dataset(ds_loader, second_half)

    train_model = model_module.model(loss=args.loss)

    output_json["model"] = {}
    output_json["model"]["details"] = train_model.info()
    output_json["model"]["details"]["loss"] = args.loss

    best_state_dict = train(train_model, train_ds, val_ds, output_json, output_name, args.epochs)

    test_model = model_module.model(pre_trained=False, loss=args.loss)
    test_model.load_state_dict(best_state_dict)

    test_threshold = get_threshold(test_model, val_ds, output_json)
    test(test_model, test_ds, test_threshold, output_json, "model")
    test(father_model, test_ds, args.threshold, output_json, "father_model")

    print("Saving best model")
    torch.save(best_state_dict, f"_models/{args.save}/{father_model_name}/{output_name}.pt")

    print("Saving result file")
    with open(f"_results/{args.save}/{father_model_name}/{output_name}.json", "w") as output:
        json.dump(output_json, output, indent=2)


def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.dataset), "Invalid dataset"
    assert 0 < args.split < 1, "Split may be between 0 and 1"
    assert args.epochs > 0, "Invalid epochs count"

    model_module = importlib.import_module(args.model)
    execute(model_module, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--father", "-f", type=str, required=True)
    parser.add_argument("--threshold", "-t", type=float, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--loss", "-l", type=str, required=True)
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--subset", "-su", type=str, required=True)
    parser.add_argument("--split", "-s", type=float, required=True)
    parser.add_argument("--epochs", "-e", type=int, required=True)
    parser.add_argument("--save", "-sa", type=str, required=True)

    main(parser.parse_args())