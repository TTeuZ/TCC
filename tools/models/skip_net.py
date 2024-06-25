import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.data_prefetcher import data_prefetcher
import torch.nn.functional as f
import torch.nn as nn
import importlib
import torch

NUM_CLASSES = 2

class skip_net(nn.Module):
    def __init__(self):
        super(skip_net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, NUM_CLASSES)


    def forward(self, x):
        x = f.relu(self.conv1(x))
        skip1 = x

        x = f.relu(self.conv2(x))
        x = x + skip1
        x = f.relu(x)
        skip2 = x

        x = f.relu(self.conv3(x))
        x = self.pool1(x)

        skip3 = self.skip_conv1(skip2)
        x = x + skip3
        x = f.relu(x)

        x = f.relu(self.conv4(x))
        x = self.pool2(x)

        skip4 = self.skip_conv2(skip3)
        x = x + skip4
        x = f.relu(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

class model():
    def __init__(self, pre_trained=False, config=None):
        self.model = skip_net()
        self.config = config

        self.model = self.model.to(self.config["device"])
        
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