from torchvision.models import MobileNet_V3_Large_Weights as pre_weights
from torchvision.models import mobilenet_v3_large as mobilenet
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.data_prefetcher import data_prefetcher
import torch

NUM_CLASSES = 2
DEVICE = "cuda"

class mobilenet_v3():
    def __init__(self, pre_trained=True):
        self.model = mobilenet(weights=pre_weights.IMAGENET1K_V2) if pre_trained else mobilenet()
        self.model.classifier[-1] = torch.nn.Linear(1280, NUM_CLASSES)
        self.model = self.model.to(DEVICE)

        self.loss = torch.nn.BCELoss()
        self.optmizer = torch.optim.Adam(self.model.parameters())
    

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
    

    def train(self, loader):
        self.model.train()
        running_loss = 0.0

        prefetcher = data_prefetcher(loader)
        inputs, labels = prefetcher.next()
        while inputs is not None:
            inputs, labels = inputs.to(DEVICE), labels.type(torch.long).to(DEVICE)
            self.optmizer.zero_grad()

            preds = self.model(inputs)
            one_hot_labels = torch.zeros(labels.size(0), 2, device=DEVICE)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

            loss = self.loss(torch.sigmoid(preds), one_hot_labels)

            loss.backward()
            self.optmizer.step()

            running_loss += loss.item() * inputs.size(0)
            inputs, labels = prefetcher.next()
        
        return running_loss / len(loader.dataset)
    

    def predict(self, loader):
        self.model.eval()

        final_labels, final_preds, final_loss = [], [], 0.0
        local_loss = torch.nn.BCELoss()

        prefetcher = data_prefetcher(loader)
        inputs, labels = prefetcher.next()
        with torch.no_grad():
            while inputs is not None:
                inputs, labels = inputs.to(DEVICE), labels.type(torch.long).to(DEVICE)

                preds = self.model(inputs)
                one_hot_labels = torch.zeros(labels.size(0), 2, device=DEVICE)
                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

                loss = local_loss(torch.sigmoid(preds), one_hot_labels)
                final_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(preds, 1)
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

                inputs, labels = prefetcher.next()
        
        final_loss = final_loss / len(loader.dataset)
        accuracy = accuracy_score(final_labels, final_preds)
        cm = confusion_matrix(final_labels, final_preds)

        return (final_loss, accuracy, cm)