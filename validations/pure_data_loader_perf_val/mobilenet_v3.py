import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torchvision.models import MobileNet_V3_Large_Weights as pre_weights
from torchvision.models import mobilenet_v3_large as mobilenet
import importlib
import torch

NUM_CLASSES = 2
DEVICE = "cuda"

class model():
    def __init__(self, pre_trained=True, loss="CrossEntropyLoss", config=None):
        self.model = mobilenet(weights=pre_weights.IMAGENET1K_V2) if pre_trained else mobilenet()

        self.model.classifier[-3] = torch.nn.Sequential(torch.nn.Linear(1280, 1024), torch.nn.ReLU(inplace=True))
        self.model.classifier[-2] = torch.nn.Sequential(torch.nn.Linear(1024, 128), torch.nn.ReLU(inplace=True))
        self.model.classifier[-1] = torch.nn.Linear(128, NUM_CLASSES)
        self.model = self.model.to(DEVICE)

        self.config = config
        if self.config["training_mode"] == "transfer":
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        loss_module = importlib.import_module("torch.nn")
        self.loss = getattr(loss_module, loss)()
        self.loss_name = loss

        self.optmizer = torch.optim.AdamW(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optmizer, gamma=0.96)

    
    def info(self):
        output = {}
        output["cuda"] = next(self.model.parameters()).is_cuda
        output["classifier_classes"] = NUM_CLASSES

        return output
    

    def get_state_dict(self):
        return self.model.state_dict()
    

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)


    def get_loss_in_dataset(self, loader):
        self.model.eval()

        loss_module = importlib.import_module("torch.nn")
        local_loss = getattr(loss_module, self.loss_name)()
        final_loss = 0.0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.type(torch.long).to(DEVICE)
                preds = self.model(inputs)

                loss = local_loss(preds, labels)
                final_loss += loss.item() * inputs.size(0)
            
        return final_loss / len(loader.dataset)


    def get_probs(self, loader):
        self.model.eval()
        final_probs, final_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.type(torch.long).to(DEVICE)
                probs = torch.sigmoid(self.model(inputs))

                final_probs.extend(probs.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
        
        return (final_probs, final_labels)  


    def predict(self, loader, threshold=0.5):
        self.model.eval()
        final_preds, final_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.type(torch.long).to(DEVICE)
                probs = torch.sigmoid(self.model(inputs))
                class_1_probs = probs[:, 1]

                predicted = (class_1_probs > threshold).long()
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
        
        return (final_preds, final_labels)  
    

    def train(self, loader):
        self.model.train()

        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.type(torch.long).to(DEVICE)
            self.optmizer.zero_grad()

            preds = self.model(inputs)
            loss = self.loss(preds, labels)

            loss.backward()
            self.optmizer.step()
        
        self.scheduler.step()