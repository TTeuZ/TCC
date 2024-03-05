# Activity
This activity ains to do a cross training using the PKLot2.0 dataset

## Dataset
The PKLot2.0 dataset was used.

## Activity steps
- Cropp all images from the new dataset (PKLot2.0 and CNRPark-EXT)
- Run a cross training using both datasets
    - 5 times training with whole PKLot2.0 and testing with whole CNRPark-EXT
    - 5 times training with whoe CNRPark-EXT and testing with whole PKLot2.0

### How to reproduce
- Create a folder called datasets in the root;
- Add the PKLot2.0 dataset there;
- Add the CNRPark-EXT images inside the PKLot2.0 folder;
- Run the generate_dataset.py script to cropp the images;
- Run the main.py script to run the MobileNet_V3.

### Notes
Paths may be changed in the code.

### Files
- \__study__: Folder containing a bunch of python notebooks to study and check each new aspect used in this experiment
- _failed: Folder containing the images that failed in the cropp phase
- _models: Folder containing saved result models
- _result: Folder containing saved json files of each experiment result
- dataset
    - data_loader: Python class used to load datasets as pytorch datasets
    - data_prefetcher: Python class used to faster GPU image loading, thus, faster training
- model
    - mobilenet_v3.py: Python class wrapping the pytorch MobileNet_v3 Implementation
    - utils.py: Python script for utils
- generate_dataset.py: Python script to cropp and save the datasets
- main.py: Main python script to run the experiment
- sumary.ipynb: Python notebook to sumarize results

