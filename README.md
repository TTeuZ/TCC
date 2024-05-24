# Checkpoint 7
## Changes

In this checkpoint we introduced the Mobilenet V3 small as an possible option, performing tests to evaluate the results in the cross testing.

Also, one major change has been made in the cross testing script, the sample generator. This class aims to create random samples from the training dataset in each epoch, trying to achieve a better/generalized result in the end.

The sample process is the following:
- For each subset:
	- For each day:
		- Split the day in morning (0am to 12am) and afternoon(12am to 12pm)
		- get N occupied and empty random samples from each time box.
- Repeat the process for all days within the training dataset.

With this process, we are able to create samples with 18450 images (CNRPark) and 16492 images (PKLot).

The labeling pipeline stayed the same.

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
- checkpoint_1: There you can find the cross testing V1 script;
- checkpoint_2: There you can find the labeling pipeline V1 script. The cross testing script stays the same;
- checkpoint_3: There you can find the labeling pipeling V2 script. The cross testing script stays the same;
- checkpoint_4: There you can the first data leveling implementation for both cross testing and labeling pipeline;
- checkpoint_5: There you can find a set of tests to determinate one best config;
- checkpoint_6: There you can find the resutls with small modifications within the dataset. No changes in the code.
