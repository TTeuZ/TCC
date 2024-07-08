# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- trained_At: /tmp/pmla20/CNRParkEXTSegmented
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
- Train days: 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 - 10 - 11 - 12 - 13 - 14
- Split: 0.75
- Sumary: _summaries/summary_9e9a4fb2-dc9a-4d7e-bb61-4bf6e7fccee5
