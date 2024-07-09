# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Models used:
    - resources/fathers/PKLot/model_691f6b7b-3f4e-4610-9060-27e012eedc88.pt
    - resources/fathers/PKLot/model_733c9384-4948-4c1a-91cc-852556ae1937.pt
    - resources/fathers/PKLot/model_878b56f2-b09c-4650-96cd-b0a1c4712fb9.pt
    - resources/fathers/PKLot/model_7808af69-b7a7-4fb6-8243-22256d3e9716.pt
    - resources/fathers/PKLot/model_c1152be4-1fe4-49bc-8767-dd46ac07bc2e.pt

## Sons
- Model: tools.models.custom_3_layer
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /tmp/pmla20/CNRParkEXTSegmented
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
- Training epocs: 20
- Train/Val split: 0.75
- Sumary: _summaries/summary_01dc926b-ce50-44c7-9cb9-6d949f8c8fc7
