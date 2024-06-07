import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import torch

ENSEMBLE_SUBPATH = "_ensembles"

class ensemble_model():
    def __init__(self, module, config, father, ensemble_exp):
        self.module = module
        self.config = config
        self.father = father.split("/")[-1].split(".")[0]
        self.ensemble_exp = ensemble_exp
        self.models = []

        ensemble_path = f"{ENSEMBLE_SUBPATH}/{self.ensemble_exp}/{self.father}"

        for model in os.listdir(ensemble_path):
            model_path = f"{ensemble_path}/{model}"

            loaded_model = self.module.model(pre_trained=False, config=self.config)
            loaded_model.load_state_dict(torch.load(model_path, map_location=self.config["device"]))
            self.models.append(loaded_model)
    

    def info(self):
        output = {}

        ensemble_path = f"{ENSEMBLE_SUBPATH}/{self.ensemble_exp}/{self.father}"
        output["models"] = []
        for model in sorted(os.listdir(ensemble_path)):
            output["models"].append(model)

        return output


    def get_probs(self, loader):
        models_probs = []

        for model in self.models:
            models_probs.append(model.get_probs(loader))
        
        labels = models_probs[0][1]
        
        all_probs = [prob[0] for prob in models_probs]
        stacked_probs = np.array(all_probs)
        final_probs = np.mean(stacked_probs, axis=0)
        final_probs = final_probs.astype(np.float32)

        return (final_probs, labels)
    

    def predict(self, loader, threshold=0.5):
        probs, labels = self.get_probs(loader)

        class_1_probs = probs[:, 1]
        preds = (class_1_probs > threshold).astype(int)

        return (preds, labels)
