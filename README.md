# TCC Scripts and Files
This repository contains all scripts and files used during this TCC. The structure is based in 2 folders: **experiments** and **main**.

The experiments folder contains a set of experiments that were used to shape the “final” scripts in the main folder. These experiments helped to set standards and also collect results that were used to improve the process.

Each experiment has its own README.md file that gives a brief introduction about the experiment process and goal.

## Datasets
The datasets used aren't stored in this repository, you will need a copy of PKLot2.0 and CNRPark-EXT to reproduce the experiments.

## Branchs
A branching strategy was chosen to allow reproducibility of each result.

Basically, if any modification was introduced in the main script that could make the result irreproducible, a “checkpoint branch” was created, keeping the script in the old state, where it is possible to reproduce the same experiment.

Long story short, the main branch can reproduce the **latest version of the results** and each checkpoint branch can reproduce a certain result in the timeline.

### Checkpoints
- old_structure: Here you can reproduce any of the testing experiments that helped shape the main script.
