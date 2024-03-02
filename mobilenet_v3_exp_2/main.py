from dataset.data_loader import data_loader
import argparse
import json
import os

# Fixed num epochs for training for now
NUM_EPOCHS = 10

def execute(args):
    train_dataset_name = args.train.split("/")[-1]
    test_dataset_name = args.test.split("/")[-1]

    ds_loader = data_loader()
    train, val = ds_loader.load_dataset(args.train, args.split)
    test = ds_loader.load_dataset(args.test)

    for index in range(args.repeat):
        output_json = {}
        output_json["dataset"] = {"train": train_dataset_name, "test": test_dataset_name, "train_val_split": args.split}



        with open(f"_results/train_{train_dataset_name}_test_{test_dataset_name}_{index}.json", "w") as output:
            json.dump(output_json, output, indent=2)


def main(args):
    assert os.path.exists(args.train), "Invalid train/val dataset"
    assert os.path.exists(args.test), "Invalid test dataset"
    assert 0 < args.split < 1, "Split may be between 0 and 1"
    assert args.repeat > 0, "Invalid repeat count"
    
    if not os.path.exists("_models"):
        os.mkdir("_models")

    if not os.path.exists("_results"):
        os.mkdir("_results")
    
    execute(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", "-tr", type=str, required=True)
    parser.add_argument("--test", "-te", type=str, required=True)
    parser.add_argument("--split", "-s", type=float, required=True)
    parser.add_argument("--repeat", "-r", type=int, required=True)

    main(parser.parse_args())

# Train with PKLot2.0 and test with CNRPark-EXT 5 times with 0.7 train size
# python3 main.py -tr /home/tteuz/Desktop/TCC/datasets/PKLot2.0/PKLotSegmented -te /home/tteuz/Desktop/TCC/datasets/PKLot2.0/CNRPartEXTSegmented -s 0.7 -r 5
    
# Train with CNRPark-EXT and test with PKLot 5 times with 0.7 train size
# python3 main.py -tr /home/tteuz/Desktop/TCC/datasets/PKLot2.0/CNRPartEXTSegmented -te /home/tteuz/Desktop/TCC/datasets/PKLot2.0/PKLotSegmented -s 0.7 -r 5