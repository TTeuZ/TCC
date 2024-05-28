# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Models used:
    - resources/fathers/CNRPark/model_5ce21bdc-67c7-4980-b3f1-685c8c78481f.pt
    - resources/fathers/CNRPark/model_7fb73074-3b6f-40e8-8aa9-3412666bdce9.pt
    - resources/fathers/CNRPark/model_75ff2c26-fe61-427e-99a8-7c978bfd0aaf.pt
    - resources/fathers/CNRPark/model_872b50d0-3d6a-434c-9026-ac7ae44eba6b.pt
    - resources/fathers/CNRPark/model_e849d4df-612f-4044-8677-76d50eff5c11.pt

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
- Sumary: _summaries/summary_2e6b8925-62b0-413e-b134-e9afeeca4a49
