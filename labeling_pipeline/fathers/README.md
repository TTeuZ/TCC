# Fathers folder
already trained models used in the labeling pipeline

## Structure
Add inside this folder another folder with your models ando also an construct.json.

You will point this folder in the run.py and the script will detect the models and configs automatically.

### construct.json
Build the json following this example

```
{
    "trained_at": "/media/tteuz/ssd/datasets/PKLot2.0/PKLotSegmented",
    "models_qty": 5,
    "models": [
        {
            "name": "model_8e9989a8-1589-44b2-8f11-41a1bbc47234.pt",
            "threshold": 0.652393639087677
        },
        {
            "name": "model_47fc46ce-4bfd-43d9-bc3a-30ebcd67633a.pt",
            "threshold": 0.514877200126648
        },
        {
            "name": "model_000433a1-27bd-4532-8241-354d89b5c854.pt",
            "threshold": 0.5845631957054138
        },
        {
            "name": "model_4388d475-d38f-455b-926f-f6cc82b894fe.pt",
            "threshold": 0.5365752577781677
        },
        {
            "name": "model_c9ca7683-15a6-43d8-be2f-3e8874c08475.pt",
            "threshold": 0.6202115416526794
        }
    ]
}
```