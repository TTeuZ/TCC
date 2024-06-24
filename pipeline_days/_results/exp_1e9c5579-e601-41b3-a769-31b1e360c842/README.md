# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- trained_At: /tmp/pmla20/CNRParkEXTSegmented
- Training mode: transfer
- Models used:
    - resources/fathers/CNRPark/model_9eab364f-4220-4cad-b7ce-11dd3fee9f93.pt
    - resources/fathers/CNRPark/model_899b2c6a-e9cd-4564-b865-7c052b74cf4f.pt
    - resources/fathers/CNRPark/model_855804d2-4891-4d9a-aa67-c6179f90b912.pt
    - resources/fathers/CNRPark/model_a05cfcd1-a513-41b2-b03e-a1fb6895f21f.pt
    - resources/fathers/CNRPark/model_d1b183f4-048c-44dc-bcf9-2151093b7bf2.pt

## Sons
- Model: tools.models.custom_3_layer
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /tmp/pmla20/PKLotSegmented
- Subsets:
    - PUCPR
    - UFPR04
    - UFPR05
- Training epocs: 60
- Train days: 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 - 10
- val days: 2
- Sumary: _summaries/summary_1e9c5579-e601-41b3-a769-31b1e360c842
