# Checkpoint 12
## Changes
In this checkpoint we modified the pipeline_days script to select the first days as validation and used a split value, like 0.75, to divide the dataset in training and validation.

Also, we added two new small models to test, the skip_net and residual_net.

## Datasets
The datasets used aren't stored in this repository, you will need a copy of PKLot2.0 and CNRPark-EXT to reproduce the experiments.

## Structure
- acr: Folder with architecture diagrams and helper files;
- cross_testing: Folder with cross testing script. More details in the readme within;
- labeling_pipeline: Folder with labeling pipeline script. More details in the readme within;
- tools: Folder with tool classes and scripts such as:
	- dataset: Script and classes that deal with datasets;
	- models: Wrappers classes for models;
	- utils: Utilities classes.

## Branchs
A branching strategy was chosen to allow reproducibility of each result.

Basically, if any modification is introduced in the main script that could make the result irreproducible, a “checkpoint branch” is created, keeping the script in the old state, where it is possible to reproduce the same experiment.

Long story short, the main branch can reproduce the **latest version of the results** and each checkpoint branch can reproduce a certain result in the timeline.

### Checkpoints
- checkpoint_0: There you can find the first experiments performed that helped shape the scripts;
- checkpoint_1: There you can find the cross testing V1 script;
- checkpoint_2: There you can find the labeling pipeline V1 script. The cross testing script stays the same;
- checkpoint_3: There you can find the labeling pipeling V2 script. The cross testing script stays the same;
- checkpoint_4: There you can the first data leveling implementation for both cross testing and labeling pipeline;
- checkpoint_5: There you can find a set of tests to determinate one best config;
- checkpoint_6: There you can find the results with small modifications within the dataset. No changes in the code;
- checkpoint_7: There you can find the first application of the sample generator in the cross testing. Also, the first MobileNetV3 Small use;
- checkpoint_8: There you can find the labeling pipeline V3 script. Also, here we have the fisrt use of the custom 3-layer CNN;
- checkpoint_9: There you can find the labeling pipeline v5 script. Also, small modifications were made in the cross testing script;
- checkpoint_10: There you can find the first ensemble implementation in the labeling pipeline;
- checkpoint_11: There you can find the first implementation of the pipeline_days and also the labeling pipeline change that allows to specify training days quantity.
