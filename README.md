# Latest Checkpoint
## Diferences between the second and third checkpoint
Here an data leveling was applied in both **cross testing** and **labeling pipeline**, leveling through the class with lowest amount.

Also, the data protocol was slightly corrected, making the split between train and val to be 70% of the first days in **each subset** and the rest (30%) to val.

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
- checkpoint_3: There you can dinf the labeling pipeling V2 script. The cross testing script stays the same.
