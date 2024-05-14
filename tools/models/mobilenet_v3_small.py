import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torchvision.models import MobileNet_V3_Small_Weights as pre_weights
from torchvision.models import mobilenet_v3_small as mobilenet
from dataset.data_prefetcher import data_prefetcher
import importlib
import torch

NUM_CLASSES = 2

class model():
    def __init__(self, pre_trained=True, config=None):
        self.model = mobilenet(weights=pre_weights.IMAGENET1K_V1) if pre_trained else mobilenet()
        self.config = config

        self.model.classifier[-1] = torch.nn.Linear(1280, NUM_CLASSES)
        self.model = self.model.to(self.config["device"])

        if self.config["training_mode"] == "transfer":
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "features.16" not in name:
                    param.requires_grad = False

        loss_module = importlib.import_module("torch.nn")
        self.loss = getattr(loss_module, self.config["loss"])()

        optimizer_module = importlib.import_module("torch.optim")
        self.optimizer = getattr(optimizer_module, self.config["optimizer"])(self.model.parameters())

    
    def info(self):
        output = {}
        output["cuda"] = next(self.model.parameters()).is_cuda
        output["classifier_classes"] = NUM_CLASSES

        return output
    

    def get_state_dict(self):
        return self.model.state_dict()
    

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.config["device"])


    def get_loss_in_dataset(self, loader):
        self.model.eval()

        loss_module = importlib.import_module("torch.nn")
        local_loss = getattr(loss_module, self.config["loss"])()
        final_loss = 0.0

        prefetcher = data_prefetcher(loader, self.config["normalize_data"])
        inputs, labels = prefetcher.next()
        with torch.no_grad():
            while inputs is not None:
                preds = self.model(inputs)

                loss = local_loss(preds, labels)
                final_loss += loss.item() * inputs.size(0)

                inputs, labels = prefetcher.next()
            
        return final_loss / len(loader.dataset)


    def get_probs(self, loader):
        self.model.eval()
        final_probs, final_labels = [], []

        prefetcher = data_prefetcher(loader, self.config["normalize_data"])
        inputs, labels = prefetcher.next()
        with torch.no_grad():
            while inputs is not None:
                probs = torch.sigmoid(self.model(inputs))

                final_probs.extend(probs.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

                inputs, labels = prefetcher.next()
        
        return (final_probs, final_labels)  


    def predict(self, loader, threshold=0.5):
        self.model.eval()
        final_preds, final_labels = [], []

        prefetcher = data_prefetcher(loader, self.config["normalize_data"])
        inputs, labels = prefetcher.next()
        with torch.no_grad():
            while inputs is not None:
                probs = torch.sigmoid(self.model(inputs))
                class_1_probs = probs[:, 1]

                predicted = (class_1_probs > threshold).long()
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

                inputs, labels = prefetcher.next()
        
        return (final_preds, final_labels)  
    

    def train(self, loader):
        self.model.train()

        prefetcher = data_prefetcher(loader, self.config["normalize_data"])
        inputs, labels = prefetcher.next()
        while inputs is not None:
            self.optimizer.zero_grad()

            preds = self.model(inputs)
            loss = self.loss(preds, labels)

            loss.backward()
            self.optimizer.step()

            inputs, labels = prefetcher.next()