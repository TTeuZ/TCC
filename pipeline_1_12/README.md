# Labeling Pipeline
## Details

This version of the pipeline aims to recreate the labeling_pipeline flow using a sequence of training days specified in the config in order to find the minimum days necessary to achieve a reasonable result in the son model.

The flow is the same as the labeling pipeline, but it runs one time for each quantity of training days specified.

This version was created seeking better performance in the experiment run.

## Diferences from the labeling pipeline
The mandatory changes in this pipeline are:
- Always, the 2 first days are for validation, the training days start counting after the first 2.
- Only the images from the 13th day and beyond are used for the test, no matter how many train days.

## Ensemble training

Each father model from the fathers array in the config gains its own ensemble. The other N models that compose the ensemble are trained in the following way:
- Given the train dataset;
- Train one model letting one camera as validation dataset.

For example, given the PKLot dataset:
- PUCPR and UFPR04 as train and UFPR05 as validation;
- PUCPR and UFPR05 as train and UFPR04 as validation;
- UFPR04 and UFPR05 as train and PUCPR as validation.

## Basic flow
- Given one father model;
- Create the ensemble;
- For each subset;
    - for each days quantity in the training days array:
        - Classify this days with the father ensemble;
            - Select only the ones that the model is more than 95% sure that the label is correct;
            - Flatten the data by the smallest class (in qty).
        - Use these images to train a new model;
            - The train model(son) has its initial weights from one model already trained in the PKLot or CNRPark;
            - Test the model before refining.
        - Use this new model to classify the other days;
        - Also classify this other days with the father model.
- Summarize all the results to compare the father model with the created one by the pipeline.

## How to Reproduce
- If necessary, crop all images from the new dataset (PKLot2.0 and CNRPark-EXT)
    - In order to do that, run the script generate_dataset.py, look into the code for more details.
- Run the script run.py

## Notes
- Paths may be changed in the code.
- Each run of the main.py creates a .json file that contains more information about that specified run.
- In order to test different models, you only need to create a wrapper class inside the models folder (follow the MobileNetV3.py class as example) and send it as the model in the run.py.

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