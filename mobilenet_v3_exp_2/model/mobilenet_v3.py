from torchvision.models import MobileNet_V3_Large_Weights as pre_weights
from torchvision.models import mobilenet_v3_large as mobilenet
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.data_prefetcher import data_prefetcher
import torch

# Fixed num classes for now
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
    

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0

        max_iters = 10
        count = 0

        prefetcher = data_prefetcher(train_loader)
        inputs, labels = prefetcher.next()
        while inputs is not None:
            if count == max_iters:
                break

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            self.optmizer.zero_grad()

            preds = self.model(inputs).squeeze(1)
            loss = self.loss(preds, labels.float())

            loss.backward()
            self.optmizer.step()

            running_loss += loss.item() * inputs.size(0)
            inputs, labels = prefetcher.next()

            count += 1
        
        return running_loss / len(train_loader.dataset)
    

    def validate(self, val_loader):
        self.model.eval()

        val_labels, val_preds, val_loss = [], [], 0.0
        bce_loss = torch.nn.BCEWithLogitsLoss()

        max_iters = 5
        count = 0

        prefetcher = data_prefetcher(val_loader)
        inputs, labels = prefetcher.next()
        with torch.no_grad():
            while inputs is not None:
                if count == max_iters:
                    break

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                preds = self.model(inputs)

                loss = bce_loss(preds.squeeze(1), labels.float())
                val_loss += loss.item() * inputs.size(0)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend((torch.sigmoid(preds).cpu().numpy() > 0.5).astype(int))

                inputs, labels = prefetcher.next()

                count += 1
        
        val_loss = val_loss / len(val_loader.dataset)
        accuracy = accuracy_score(val_labels, val_preds)
        cm = confusion_matrix(val_labels, val_preds)

        return (val_loss, accuracy, cm)