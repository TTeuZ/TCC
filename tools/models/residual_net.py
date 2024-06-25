import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.data_prefetcher import data_prefetcher
import torch.nn.functional as f
import torch.nn as nn
import importlib
import torch

NUM_CLASSES = 2

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.shortcut(x)
        x = f.relu(x)
        return x


class residual_net(nn.Module):
    def __init__(self):
        super(residual_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.resblock1 = residual_block(32, 32)
        self.resblock2 = residual_block(32, 64, stride=2)
        self.resblock3 = residual_block(64, 128, stride=2)

        self.fc = nn.Linear(512, NUM_CLASSES)


    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class model():
    def __init__(self, pre_trained=False, config=None):
        self.model = residual_net()
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