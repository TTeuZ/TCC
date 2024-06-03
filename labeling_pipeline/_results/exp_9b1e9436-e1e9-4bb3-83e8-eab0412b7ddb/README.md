# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: normal
- Models used:
    - resources/fathers/CNRPark/model_16ea94a6-dba6-4316-8ca9-11de6b76748f.pt
    - resources/fathers/CNRPark/model_877a70fc-8c37-4127-a622-1ef10c249a5b.pt
    - resources/fathers/CNRPark/model_afc23fb2-10b5-481d-bbaa-20dc8c4da998.pt
    - resources/fathers/CNRPark/model_bca64c19-f517-4edf-aeef-e298ceda7d48.pt
    - resources/fathers/CNRPark/model_e03c6d97-c83a-4465-9f59-207ff5463822.pt

## Sons
- Model: tools.models.custom_3_layer
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /media/tteuz/ssd/datasets/PKLot2.0/PKLotSegmented
- Subsets:
    - PUCPR
    - UFPR04
    - UFPR05
- Training epocs: 15
- Train/Val split: 0.7
- Sumary: _summaries/summary_9b1e9436-e1e9-4bb3-83e8-eab0412b7ddb
