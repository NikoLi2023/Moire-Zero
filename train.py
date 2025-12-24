import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
import datasets
from models import Pipeline,  MZNetLocal

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training MZNet')
    parser.add_argument("--config", default='UHDM.yml', type=str,
                        help="Path to the config file")
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


    # # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> Using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)

    # create model
    print("=> Creating model...")
    pipeline = Pipeline(args, config)   
    print(sum(p.numel() for p in pipeline.model.parameters()))
    pipeline.train(DATASET)


if __name__ == "__main__":
    main()

