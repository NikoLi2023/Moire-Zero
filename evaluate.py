import argparse
import os
import yaml
import torch
import numpy as np
import datasets
from models import Pipeline
from pathlib import Path


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate MZNet')
    parser.add_argument("--config", default='UHDM.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--seed', default=42, type=int, metavar='N',
                        help='Seed for initializing training (default: 42)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> Using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # create model
    print("=> Creating model")
    pipeline = Pipeline(args, config)
    Path(config.eval.result_folder).mkdir(parents=True, exist_ok=True)

    pipeline.eval(val_loader)


if __name__ == '__main__':
    main()
