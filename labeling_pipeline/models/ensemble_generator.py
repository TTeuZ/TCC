import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cross_testing.dataset.data_leveler import data_leveler
from tools.utils.utils import create_folder
from dataset.data_loader import data_loader
import importlib
import torch
import math
import copy

class ensemble_generator():
    def __init__(self, config, exp_name):
        self.exp_name = exp_name

        self.model_config = config["fathers"]["config"]

        self.dl_config = config["fathers"]["dl_config"]
        self.dl_config = {key: self.dl_config.get(key, config["experiment"].get(key)) for key in ["img_size", "batch_size", "num_workers"]}

        self.epochs = config["experiment"]["epochs"]
        self.dataset = config["fathers"]["trained_at"]

        self.model_module = importlib.import_module(config["fathers"]["module"])

        create_folder("_ensembles")
        create_folder(f"_ensembles/{self.exp_name}")


    def _train(self, model, train_ds, val_ds, epochs, dl_config):
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=dl_config["num_workers"], pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=dl_config["batch_size"], shuffle=False, num_workers=dl_config["num_workers"], pin_memory=True)
        best_state_dict, best_loss = None, math.inf

        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}] Initializing training epoch", end=" ")

            model.train(train_loader)
            val_loss = model.get_loss_in_dataset(val_loader)

            if (best_loss - val_loss) > 0.0:
                best_state_dict = copy.deepcopy(model.get_state_dict())
                best_loss = val_loss

            print(f" loss: {val_loss} - best_loss: {best_loss}")

        return best_state_dict


    def generate(self, father):
        save_path = f"_ensembles/{self.exp_name}/{father.split('/')[-1].split('.')[0]}"
        create_folder(save_path)

        ds_loader = data_loader(self.dl_config)
        ds_leveler = data_leveler()
        os.system(f"cp {father} {save_path}")

        subsets = sorted(os.listdir(self.dataset))
        for subset in subsets:
            train_ds, val_ds = ds_loader.load_dataset_with_val_subset(self.dataset, subset)
            flattened_train_ds = ds_leveler.flatten_dataset(train_ds)
            flattened_val_ds = ds_leveler.flatten_dataset(val_ds)

            print(f"training ensemble model {subset}...")
            train_model = self.model_module.model(config=self.model_config)
            best_state_dict = self._train(train_model, flattened_train_ds, flattened_val_ds, self.epochs, self.dl_config)

            torch.save(best_state_dict, f"{save_path}/{subset}.pt")