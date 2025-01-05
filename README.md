# Latest Checkpoint
This is the latest and last checkpoint of this work. Here you can find the results that were presented in the published paper.

## Paper
- Title: Optimizing Parking Space Classification: Distilling Ensembles into Lightweight Classifiers
- Conference: ICMLA 2024, Miami - Florida
- Link: https://arxiv.org/abs/2410.14705

## Experiments

Each folder represents one of the experiments held in this research.

The **cross_testing** folder contains the code used to train the Teachers (Father) models and to pre-train the students (Son) models.

All folders that contains the word **pipeline** were used to test and collect results of the proposed workflow (better described in the paper).
- labeling_pipeline contains the code that runs the pipeline using n days of training
- pipeline_classify contains the code that test different teachers a posteriori thresholds to collect pseudo-labels
- pipeline_days contains the code that test different days quantity to collect pseudo-labels
- pipeline_raspberry contains the code to run and test the students in one raspberry pi
- pipeline_true_labels contains the code to run the pipeline with the ground truth

## Branchs
A branching strategy was chosen to allow reproducibility of each result.

Basically, if any modification is introduced in the main script that could make the result irreproducible, a “checkpoint branch” is created, keeping the script in the old state, where it is possible to reproduce the same experiment.

Long story short, the main branch can reproduce the **latest version of the results** (relevant ones) and each checkpoint branch can reproduce a certain result in the timeline (just for keep track and control while the research was ongoing).

### Checkpoints
- checkpoint_0: There you can find the first experiments performed that helped shape the scripts;
- checkpoint_1: There you can find the cross testing V1 script;
- checkpoint_2: There you can find the labeling pipeline V1 script. The cross testing script stays the same;
- checkpoint_3: There you can find the labeling pipeling V2 script. The cross testing script stays the same;
- checkpoint_4: There you can find the first data leveling implementation for both cross testing and labeling pipeline;
- checkpoint_5: There you can find a set of tests to determinate one best config;
- checkpoint_6: There you can find the results with small modifications within the dataset. No changes in the code;
- checkpoint_7: There you can find the first application of the sample generator in the cross testing. Also, the first MobileNetV3 Small use;
- checkpoint_8: There you can find the labeling pipeline V3 script. Also, here we have the fisrt use of the custom 3-layer CNN;
- checkpoint_9: There you can find the labeling pipeline v5 script. Also, small modifications were made in the cross testing script;
- checkpoint_10: There you can find the first ensemble implementation in the labeling pipeline;
- checkpoint_11: There you can find the first implementation of the pipeline_days and also the labeling pipeline change that allows to specify training days quantity;
- checkpoint_12: There you can find the pipeline_days second version script, that modifies a little the train/val split, and also the first tests with the residual_net and skip_net;
- checkpoint_13: There you can find the first results with the new dataset and the old train/val split criteria.

# Note
The terms Father and Son were used in the code, which differs from the terms used in the paper (Teacher and Student). The correct is use Teacher and Student, but, for the sake of sanity, the code still uses the terms father and son =D.