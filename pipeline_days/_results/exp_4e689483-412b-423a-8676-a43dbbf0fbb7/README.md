# Experiment infos

## Fathers
- Model: tools.models.mobilenet_v3_large
- Training mode: transfer
- Trained at: /tmp/pmla20/CNRParkEXTSegmented
- Models used:
    - model_b52709b5-e9ec-43f8-b446-a22471c743be.pt
    - model_f7f694d6-62f3-4d41-91a7-39f40d664919.pt
    - model_58e35bb0-e5eb-4bad-91b2-34efc75f674f.pt
    - model_5c7500bf-70db-450b-927c-6c397a8aa42e.pt
    - model_cc1bdd1b-5388-4951-9f62-2e22824c2656.pt

## Sons
- Model: tools.models.residual_net
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Experiment
- Dataset: /tmp/pmla20/PKLotSegmented
- Subsets:
    - PUCPR
    - UFPR04
    - UFPR05
- Training epocs: 20
- Train days: 6 - 7 - 8 - 9 - 10 - 11 - 12 - 13 - 14
- Split: 0.75
- Sumary: _summaries/summary_4e689483-412b-423a-8676-a43dbbf0fbb7
