# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Models used:
    - resources/fathers/PKLot/model_01f92b59-2892-470a-86c6-3d03d6f8f115.pt
    - resources/fathers/PKLot/model_2ad1524e-fd50-4ca6-89ff-5fd3e239a64f.pt
    - resources/fathers/PKLot/model_461ac9ae-b90a-4f54-a8f7-e6087903c182.pt
    - resources/fathers/PKLot/model_851e55fb-c950-4e3d-8682-27f554afe43b.pt
    - resources/fathers/PKLot/model_8761c3eb-daa4-4ecc-a060-68faf3d36f22.pt

## Sons
- Model: tools.models.mobilenet_v3_large
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /media/tteuz/ssd/datasets/PKLot2.0/CNRParkEXTSegmented
- Subsets:
    - CNR-CAMERA-1
    - CNR-CAMERA-2
    - CNR-CAMERA-3
    - CNR-CAMERA-4
    - CNR-CAMERA-5
    - CNR-CAMERA-6
    - CNR-CAMERA-7
    - CNR-CAMERA-8
    - CNR-CAMERA-9
- Training epocs: 50
- Train/Val split: 0.7
- Sumary: _summaries/summary_8199a1cb-76ff-4684-8794-c61c99e55ff1
