# Checkpoint 5
## Changes
In this checkpoint, the following changes were made:
Dataset crop were re-made, trying to remove “failed” images that were with “black triangles” in the corners;
Fast collate function remove, replaced by transform.ToTensor().

With that, this checkpoint aimed to test a bunch of combinations between, model classification layer, optimizers such Adam or AdamW and learning rate decays.

In the end, the best result came from the model with the simple classification layer (removing only the last layer to output 2 classes), using Adam and without learning rate decay.

And also, both cross testing and labeling pipeline were refactored to start using configs json files to facilitate the experiment run.

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
- checkpoint_1: There you can find the cross testing V1 script.
- checkpoint_2: There you can find the labeling pipeline V1 script. The cross testing script stays the same.
- checkpoint_3: There you can find the labeling pipeling V2 script. The cross testing script stays the same.
- checkpoint_4: There you can the first data leveling implementation for both cross testing and labeling pipeline
