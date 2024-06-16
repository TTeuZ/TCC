# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Models used:
    - resources/fathers/CNRPark/model_9eab364f-4220-4cad-b7ce-11dd3fee9f93.pt
    - resources/fathers/CNRPark/model_899b2c6a-e9cd-4564-b865-7c052b74cf4f.pt
    - resources/fathers/CNRPark/model_855804d2-4891-4d9a-aa67-c6179f90b912.pt
    - resources/fathers/CNRPark/model_a05cfcd1-a513-41b2-b03e-a1fb6895f21f.pt
    - resources/fathers/CNRPark/model_d1b183f4-048c-44dc-bcf9-2151093b7bf2.pt

## Sons
- Model: tools.models.mobilenet_v3_large
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /tmp/pmla20/PKLotSegmented
- Subsets:
    - PUCPR
    - UFPR04
    - UFPR05
- Training epocs: 60
- Train/Val split: 0.7
- Sumary: _summaries/summary_bad2d29f-63d1-40d0-822f-77fede72d8e3
