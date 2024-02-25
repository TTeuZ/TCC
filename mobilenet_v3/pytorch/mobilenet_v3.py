import torch
from torchvision.models import mobilenet_v3_large as mobilenet
from torchvision.models import MobileNet_V3_Large_Weights as pre_weights

from sklearn.metrics import confusion_matrix, accuracy_score

import helpers.data_handler as data_handler
import helpers.utils as utils

if __name__ == "__main__":
    NUM_CLASSES = 1
    NUM_EPOCHS = 1

    print(f"PyTorch version: {torch.__version__}")

    print("--------------------------------------------------")
    print(f"Using cuda: {torch.cuda.is_available()}")
    print(f"Cuda corrent device: {torch.cuda.current_device()}")
    print(f"Cuda device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Torch Backend enable: {torch.backends.cudnn.enabled}")
    print(f"Torch Backend: {torch.backends.cudnn.version() }")

    train, validation, test = data_handler.get_datasets()
    print(f"Train dataset size: {len(train)}")
    print(f"validation dataset size: {len(validation)}")
    print(f"test dataset size: {len(test)}")

    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=1500, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1500, shuffle=False, num_workers=4, pin_memory=True)

    model = mobilenet(weights=pre_weights.IMAGENET1K_V2)
    model.classifier[-1] = torch.nn.Linear(1280, NUM_CLASSES)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model.classifier)
    print(f"is in cuda: {next(model.parameters()).is_cuda}")
    print(f"device: {device}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        warmup_iters = 10
        count = 0


        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if count == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
            if count >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(count))

            if count >= warmup_iters: torch.cuda.nvtx.range_push("forward")
            preds = model(inputs).squeeze(1)
            if count >= warmup_iters: torch.cuda.nvtx.range_pop()

            loss = bce_loss(preds, labels.float())

            if count >= warmup_iters: torch.cuda.nvtx.range_push("backward")
            loss.backward()
            if count >= warmup_iters: torch.cuda.nvtx.range_pop()

            if count >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
            optimizer.step()
            if count >= warmup_iters: torch.cuda.nvtx.range_pop()

            if count >= warmup_iters: torch.cuda.nvtx.range_pop()
            
            running_loss += loss.item() * inputs.size(0)
            count += 1
        
        epoch_loss = running_loss / len(train)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}")

        # Validation 
        model.eval()

        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend((torch.sigmoid(preds) > 0.5).cpu().numpy().astype(int))

        accuracy = accuracy_score(val_labels, val_preds)
        cm = confusion_matrix(val_labels, val_preds)
        
        print(f'Validation Accuracy: {accuracy:.2f}%')
        utils.print_confusion_matrix(cm)


    model.eval()

    test_labels = []
    test_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend((torch.sigmoid(preds) > 0.5).cpu().numpy().astype(int))

    accuracy = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print(f'Test Accuracy: {accuracy:.2f}%')
    utils.print_confusion_matrix(cm)