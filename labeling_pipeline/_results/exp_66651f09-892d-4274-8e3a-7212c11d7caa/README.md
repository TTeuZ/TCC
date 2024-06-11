# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: normal
- Models used:
    - resources/fathers/PKLot/model_8cc44d5d-2b0b-4507-a69c-dd8cb881567f.pt
    - resources/fathers/PKLot/model_18d7d09a-bc3a-4244-a8f4-169ecdcb60eb.pt
    - resources/fathers/PKLot/model_c356465a-388b-459d-8faf-ce38afe4ba83.pt
    - resources/fathers/PKLot/model_ccf7addb-e034-41b8-bad1-6f1278bc4644.pt
    - resources/fathers/PKLot/model_e3f3f7fb-9c8c-4d88-9c98-935b29d380b8.pt

## Sons
- Model: tools.models.custom_3_layer
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
- Training epocs: 15
- Train/Val split: 0.7
- Sumary: _summaries/summary_ba9fbbeb-07a4-422c-ab83-ba580cf0baf8
