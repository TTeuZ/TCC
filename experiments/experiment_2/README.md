# Activity
This activity aims to cross training using the PKLot2.0 and CNRPark-EXT dataset

## Dataset
The PKLot2.0 and CNRPArk-EXT dataset were used.

## Activity steps
- Crop all images from the new dataset (PKLot2.0 and CNRPark-EXT)
- Run a cross training using both datasets
    - 5 times training with whole PKLot2.0 and testing with whole CNRPark-EXT
    - 5 times training with whole CNRPark-EXT and testing with whole PKLot2.0

### How to reproduce
- Create a folder called datasets in the root;
- Add the PKLot2.0 dataset there;
- Add the CNRPark-EXT images inside the PKLot2.0 folder;
- Run the generate_dataset.py script to crop the images;
- Run the main.py script to run the MobileNet_V3.

### Notes
Paths may be changed in the code.

The folder structure to correct crop the images should be like this:

```
PKLot2.0
├── CNRPark-EXT
│   ├── cnr-camera-1_spots.json
│   ├── cnr-camera-2_spots.json
│   ├── cnr-camera-3_spots.json
│   ├── cnr-camera-4_spots.json
│   ├── cnr-camera-5_spots.json
│   ├── cnr-camera-6_spots.json
│   ├── cnr-camera-7_spots.json
│   ├── cnr-camera-8_spots.json
│   ├── cnr-camera-9_spots.json
│   ├── OVERCAST
│   ├── RAINY
│   └── SUNNY
├── PKLot
    ├── PUCPR
    ├── pucpr_spots.json
    ├── UFPR04
    ├── ufpr04_spots.json
    ├── UFPR05
    └── ufpr05_spots.json
```

Only the folders with the full images and the json (from PKLot2.0 spots) may be in the same folder

### Files
- \__study__: Folder containing a bunch of python notebooks to study and check each new aspect used in this experiment
- _failed: Folder containing the images that failed in the crop phase
- _models: Folder containing saved result models
- _result: Folder containing saved json files of each experiment result
- dataset
    - data_loader: Python class used to load datasets as pytorch datasets
    - data_prefetcher: Python class used to faster GPU image loading, thus, faster training
- model
    - mobilenet_v3.py: Python class wrapping the pytorch MobileNet_v3 Implementation
    - utils.py: Python script for utils
- generate_dataset.py: Python script to crop and save the datasets
- main.py: Main python script to run the experiment
- sumary.ipynb: Python notebook to summarize results


