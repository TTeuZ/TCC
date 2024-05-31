import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.customs_datasets import pipeline_dataset, mutable_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from tools.utils.metrics import get_roc_auc, get_eer
from dataset.data_leveler import data_leveler
from dataset.data_loader import data_loader
from torchvision import transforms
from collections import Counter
import numpy as np
import importlib
import argparse
import torch
import math
import copy
import json

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def change_transform(dataset, dl_config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(dl_config["img_size"], antialias=True)])

    for subset in dataset.datasets:
        subset.transform = transform


def divide_dataset(dataset, split=0.5):
    divisor = math.ceil(len(dataset) * split)
    return (list(dataset.items())[:divisor], list(dataset.items())[divisor:])


def get_train_val_datasets(first_half, new_labels, dl_config, split, output_json):
    divisor = math.ceil(len(first_half) * split)

    train_ds, train_labels = first_half[:divisor], new_labels[:divisor]
    val_ds, val_labels = first_half[divisor:], new_labels[divisor:]

    train_ds, train_labels = [date[1] for date in train_ds], [label[1] for label in train_labels]
    val_ds, val_labels = [date[1] for date in val_ds], [label[1] for label in val_labels]

    train_ds, train_labels = torch.utils.data.ConcatDataset(train_ds), np.concatenate(train_labels)
    val_ds, val_labels = torch.utils.data.ConcatDataset(val_ds), np.concatenate(val_labels)

    class_count = Counter(train_labels)
    before_class_0, before_class_1 = class_count[0], class_count[1]

    ds_leveler = data_leveler()
    train_ds, train_labels = ds_leveler.flatten_dataset(train_ds, train_labels)
    val_ds, val_labels = ds_leveler.flatten_dataset(val_ds, val_labels)

    class_count = Counter(train_labels)
    after_class_0, after_class_1 = class_count[0], class_count[1]

    output_json["dataset"]["train_labels"] = {"before_leveling": {"empty": before_class_0, "occupied": before_class_1},
                                              "after_leveling": {"empty": after_class_0, "occupied": after_class_1}}

    change_transform(train_ds, dl_config), change_transform(val_ds, dl_config)
    return pipeline_dataset(train_ds, train_labels), pipeline_dataset(val_ds, val_labels)


def get_test_dataset(second_half):
    test_ds = [date[1] for date in second_half]
    return mutable_dataset(torch.utils.data.ConcatDataset(test_ds))

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def test(model, test_ds, dl_config, threshold, output_json, output_model, output_type):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(dl_config["img_size"], antialias=True)])
    test_ds.change_transform(transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)

    print(f"Testing model - {output_model} - {output_type}")
    preds, labels = model.predict(test_loader, threshold)
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    output_json[output_model][output_type] = {"accuracy": accuracy, "cm": str(cm)}


def get_threshold(model, val_ds, dl_config, output_json):
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)

    probs, labels = model.get_probs(val_loader)
    probs = np.array([prob[1] for prob in probs])

    roc, auc = get_roc_auc(probs, labels)
    threshold, eer = get_eer(roc)

    output_json["model"]["metrics"] = {}
    output_json["model"]["metrics"]["auc"] = float(auc)
    output_json["model"]["metrics"]["eer"] = float(eer)
    output_json["model"]["metrics"]["threshold"] = float(threshold)

    return threshold


def train(model, train_ds, val_ds, epochs, dl_config, output_json, output_name):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=dl_config["num_workers"], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)
    best_state_dict, best_loss, epoch_id = None, math.inf, -1

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
        
        output_json["model"]["epochs"][f"epoch_{epoch + 1}"] = val_loss
    
    output_json["model"]["best_model"] = {"loss": best_loss, "epoch_id": epoch_id, "model": f"{output_name}.pt"}

    return best_state_dict


def classify(model, dl_config, first_half, output_json):
    print("Classifing images")

    all_preds, all_images, used_images, wrong_labels = [], 0, 0, 0
    for date in first_half:
        classify_loader = torch.utils.data.DataLoader(date[1], batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)
        probs, labels = model.get_probs(classify_loader)

        preds, index_to_remove = [], []
        for index, prob in enumerate(probs):
            if prob[0] > 0.9 or prob[1] > 0.9:
                preds.append(np.argmax(prob))
                if np.argmax(prob) != labels[index]: wrong_labels += 1
            else:
                index_to_remove.append(index)
        
        for index in sorted(index_to_remove, reverse=True):
            del date[1].imgs[index]
        date[1].samples = date[1].imgs

        all_images += len(probs)
        used_images += len(preds)
        all_preds.append((date[0], preds))

    output_json["dataset"]["classify"] = {"all_images": all_images, "used_images": used_images, "wrong_labels": wrong_labels}

    return all_preds


def execute(father_module, son_module, config, args):
    father_model_name = args.father.split('/')[-1][:-3]
    output_name = f"model_{args.subset}"
    output_json = {}

    father_dl_config, father_config = config["fathers"]["dl_config"], config["fathers"]["config"]
    son_dl_config, son_config = config["model"]["dl_config"], config["model"]["config"]

    selected_fields = ["img_size", "batch_size", "num_workers"]
    father_dl_config = { key: father_dl_config.get(key, config["experiment"].get(key)) for key in selected_fields }
    son_dl_config = { key: son_dl_config.get(key, config["experiment"].get(key)) for key in selected_fields }

    torch.cuda.set_device(father_config["device"])

    print(f"Starting Labeling Pipeline [FATHER: {father_model_name}][DATASET: {config['dataset']['path'].split('/')[-1]}][SUBSET: {args.subset}]")

    ds_loader = data_loader(father_dl_config)
    dataset = ds_loader.get_subset_from_dataset(config["dataset"]["path"], args.subset)
    first_half, second_half = divide_dataset(dataset)

    father_model = father_module.model(pre_trained=False, config=father_config)
    father_model.load_state_dict(torch.load(args.father))

    output_json["system_info"] = {"pytorch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())}
    output_json["dataset"] = {"dataset": config["dataset"]["path"].split('/')[-1], "subset": args.subset, "train_val_split": config["dataset"]["split"]}
    output_json["father_model"] = father_model.info()
    output_json["father_model"]["name"] = father_model_name
    output_json["father_model"]["threshold"] = args.father_threshold

    new_labels = classify(father_model, father_dl_config, first_half, output_json)
    train_ds, val_ds = get_train_val_datasets(first_half, new_labels, son_dl_config, config["dataset"]["split"], output_json)
    test_ds = get_test_dataset(second_half)

    train_model = son_module.model(pre_trained=False, config=son_config)
    train_model.load_state_dict(torch.load(args.initial))

    output_json["model"] = {}
    output_json["model"]["initial_weights"] = args.initial
    output_json["model"]["details"] = son_config

    test(train_model, test_ds, son_dl_config, args.initial_threshold, output_json, "model", "pre_refinement")

    best_state_dict = train(train_model, train_ds, val_ds, config["experiment"]["epochs"], son_dl_config, output_json, output_name)
    test_model = son_module.model(pre_trained=False, config=son_config)
    test_model.load_state_dict(best_state_dict)

    test_threshold = get_threshold(test_model, val_ds, son_dl_config, output_json)
    test(test_model, test_ds, son_dl_config, test_threshold, output_json, "model", "pos_refinement")
    test(father_model, test_ds, father_dl_config, args.father_threshold, output_json, "father_model", "test")

    print("Saving best model")
    torch.save(best_state_dict, f"_models/{args.save}/{father_model_name}/{output_name}.pt")

    print("Saving result file")
    with open(f"_results/{args.save}/{father_model_name}/{output_name}.json", "w") as output:
        json.dump(output_json, output, indent=2)


def main(args):
    assert torch.cuda.is_available(), "Cuda unavailable"
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    father_module = importlib.import_module(config["fathers"]["module"])
    son_module = importlib.import_module(config["model"]["module"])

    execute(father_module, son_module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--father", "-f", type=str, required=True)
    parser.add_argument("--father_threshold", "-ft", type=float, required=True)
    parser.add_argument("--initial", "-i", type=str, required=True)
    parser.add_argument("--initial_threshold", "-it", type=float, required=True)
    parser.add_argument("--subset", "-su", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--save", "-sa", type=str, required=True)

    main(parser.parse_args())