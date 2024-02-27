import torch
from torchvision.models import mobilenet_v3_large as mobilenet
from torchvision.models import MobileNet_V3_Large_Weights as pre_weights

import helpers.data_prefetcher as data_prefetcher
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

    collate_fn = lambda batch: utils.fast_collate(batch)

    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False, num_workers=6, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=1500, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1500, shuffle=False)

    model = mobilenet(weights=pre_weights.IMAGENET1K_V2)
    model.classifier[-1] = torch.nn.Linear(1280, NUM_CLASSES)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model.classifier)
    print(f"is in cuda: {next(model.parameters()).is_cuda}")
    print(f"device: {device}")

    num_iters = 100
    count = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        prefetcher = data_prefetcher.data_prefetcher(train_loader)

        inputs, labels = prefetcher.next()
        while inputs is not None:
            if count == num_iters:
                break

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            preds = model(inputs).squeeze(1)
            loss = bce_loss(preds, labels.float())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            count += 1

            inputs, labels = prefetcher.next()
        
        epoch_loss = running_loss / len(train)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}")