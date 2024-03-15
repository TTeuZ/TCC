# Activity
This activity aims to correct finetune an MobileNet_v3 with the PKLot dataset.

## Dataset
The PKLot dataset was used.

## Activity steps
2 implementations were tested, **Keras** and **PyTorch**.
The keras implementation was quickly discarded, due to the cleaner code in PyTorch.

### How to reproduce
- Create a folder called datasets in the root;
- Add the PKLot dataset there;
- If necessary, crop the parking slots in to the PKLotSegmented folder.
- Run any mobilenet_v3.ipynb scripts (pytorch or keras)

### Notes
Paths may be changed in the code.

### Files
- keras
    - protocol.ipynb: Python notebook with dataset split protocol
    - mobilenet_v3.ipynb: Python notebook with the keras MobileNet_v3 finetuning
- pytorch
    - protocol.ipynb: Python notebook with dataset split protocol
    - mobilenet_v3.ipynb: Python notebook with the keras MobileNet_v3 finetuning
    - bottlenect_test.py: Python script used to run the PyTOrch bottleneck to verify performance issues
    - helpers/data_handler.py: Python script to generate the pytorch dataset
    - helpers/data_prefetcher: Python class used to faster GPU image loading, thus, faster training
    - helpers/utils.py: Python script for utils



