# Latest Checkpoint
This chekopoint contains both the **cross testing** script and the first version of the **labeling pipeline**.

The cross testing performs a cross testing process with both PKlot2.0 and CNRPark-EXT and summarizes the results to get a clear view.

The labeling pipeline aims to be a full automatic process to classify images and use this previous classified images to train other models.

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

Basically, if any modification was introduced in the main script that could make the result irreproducible, a “checkpoint branch” was created, keeping the script in the old state, where it is possible to reproduce the same experiment.

Long story short, the main branch can reproduce the **latest version of the results** and each checkpoint branch can reproduce a certain result in the timeline.

### Checkpoints
- checkpoint_0: There you can find the first experiments performed that helped shape the scripts;
- checkpoint_1: There you can find the first version of the cross testing script.
