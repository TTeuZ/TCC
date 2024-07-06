# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Models used:
    - resources/fathers/CNRPark/model_0fd295f2-4629-4fd3-ba78-860127fae815.pt
    - resources/fathers/CNRPark/model_5b19aa32-a859-41e4-8a96-27b824d5fb40.pt
    - resources/fathers/CNRPark/model_92d6d304-71d2-46dc-8b22-461610a3dc2d.pt
    - resources/fathers/CNRPark/model_441eacbf-4e28-43fc-85b3-8f6cd6add7f9.pt
    - resources/fathers/CNRPark/model_26149e3a-dd5b-40c7-9f89-1b8489b37f2f.pt

## Sons
- Model: tools.models.mobilenet_v3_small
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /tmp/pmla20/PKLotSegmented
- Subsets:
    - PUCPR
    - UFPR04
    - UFPR05
- Training epocs: 20
- Train/Val split: 0.75
- Sumary: _summaries/summary_0d41d2ff-af98-4b76-889a-ea3f21722161
