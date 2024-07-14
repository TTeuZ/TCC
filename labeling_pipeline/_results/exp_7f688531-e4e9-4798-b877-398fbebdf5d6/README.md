# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Trained at: /tmp/pmla20/PKLotSegmented
- Models used:
    - model_fc601bf2-e3ab-4bc0-9629-3208e9ab42e5.pt
    - model_d13b9a03-fce4-458a-a44a-8bbaf42e5ee9.pt
    - model_e0c19f49-dea1-4aeb-bec0-83b6e2495d3f.pt
    - model_ea713e4b-f032-41f1-83ba-3f52a58b83cd.pt
    - model_6ba195b2-dcf3-49c8-b5d1-0f721e586b11.pt

## Sons
- Model: tools.models.mobilenet_v3_small
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
- Sumary: _summaries/summary_7f688531-e4e9-4798-b877-398fbebdf5d6
