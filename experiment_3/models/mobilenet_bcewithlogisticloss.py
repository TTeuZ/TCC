from torchvision.models import MobileNet_V3_Large_Weights as pre_weights
from torchvision.models import mobilenet_v3_large as mobilenet
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.data_prefetcher import data_prefetcher
import torch

NUM_CLASSES = 1
DEVICE = "cuda"

class mobilenet_v3():
    def __init__(self, pre_trained=True):
        self.model = mobilenet(weights=pre_weights.IMAGENET1K_V2) if pre_trained else mobilenet()
        self.model.classifier[-1] = torch.nn.Linear(1280, NUM_CLASSES)
        self.model = self.model.to(DEVICE)

        self.loss = torch.nn.BCEWithLogitsLoss()
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
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            self.optmizer.zero_grad()

            preds = self.model(inputs).squeeze(1)
            loss = self.loss(preds, labels.float())

            loss.backward()
            self.optmizer.step()

            running_loss += loss.item() * inputs.size(0)
            inputs, labels = prefetcher.next()
        
        return running_loss / len(loader.dataset)
    

    def predict(self, loader):
        self.model.eval()

        final_labels, final_preds, final_loss = [], [], 0.0
        local_loss = torch.nn.BCEWithLogitsLoss()

        prefetcher = data_prefetcher(loader)
        inputs, labels = prefetcher.next()
        with torch.no_grad():
            while inputs is not None:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                preds = self.model(inputs)

                loss = local_loss(preds.squeeze(1), labels.float())
                final_loss += loss.item() * inputs.size(0)

                final_labels.extend(labels.cpu().numpy())
                final_preds.extend((torch.sigmoid(preds).cpu().numpy() > 0.5).astype(int))

                inputs, labels = prefetcher.next()
        
        final_loss = final_loss / len(loader.dataset)
        accuracy = accuracy_score(final_labels, final_preds)
        cm = confusion_matrix(final_labels, final_preds)

        return (final_loss, accuracy, cm)