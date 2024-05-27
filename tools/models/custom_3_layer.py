import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.data_prefetcher import data_prefetcher
import torch.nn.functional as f
import torch.nn as nn
import importlib
import torch

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 2

class custom_3_layer(nn.Module):
    def __init__(self):
        super(custom_3_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=INPUT_SHAPE[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()

        self._to_linear = None
        self.convs(torch.randn(1, *INPUT_SHAPE))

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)


    def convs(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.pool2(x)
        x = f.relu(self.conv3(x))
        
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).size(1)
        
        return x
    

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

class model():
    def __init__(self, pre_trained=False, config=None):
        self.model = custom_3_layer()
        self.config = config

        self.model = self.model.to(self.config["device"])

        if self.config["training_mode"] == "transfer":
            for name, param in self.model.named_parameters():
                if "conv3" not in name and "fc2" not in name and "fc1" not in name:
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