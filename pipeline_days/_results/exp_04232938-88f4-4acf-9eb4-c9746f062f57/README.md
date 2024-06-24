# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- trained_At: /tmp/pmla20/PKLotSegmented
- Training mode: transfer
- Models used:
    - resources/fathers/PKLot/model_5710a2ff-c157-4389-a8c5-c8e3141e5aee.pt
    - resources/fathers/PKLot/model_8909415f-6afe-41ab-9ac7-350b8d621ad7.pt
    - resources/fathers/PKLot/model_80645867-bf7e-4678-966d-be20067b1016.pt
    - resources/fathers/PKLot/model_a4f31131-fb3a-4a64-9610-9e8ad2b8812d.pt
    - resources/fathers/PKLot/model_b6254325-9f25-40cb-8dd6-f7a20a586533.pt

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
- Training epocs: 60
- Train days: 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 - 10
- val days: 2
- Sumary: _summaries/summary_04232938-88f4-4acf-9eb4-c9746f062f57
