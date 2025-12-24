<div align="center">
<h1 align="center">
Moiré Zero: An Efficient and High-Performance <br>
Neural Architecture for Moiré Removal
</h1>


[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://sngryonglee.github.io/MoireZero/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16973-b31b1b.svg)](https://www.arxiv.org/abs/2507.22407)
</div>

Official github for "**Moiré Zero: An Efficient and High-Performance Neural Architecture for Moiré Removal**"

<img src="figures/architecture.jpg">

## Installation

```bash
# create conda environment
conda create -n mznet python=3.9 -y
conda activate mznet

# install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train

Create a `metadata.csv` file for your dataset and specify its path in the config file.

Specify the configuration file that matches your target dataset and model.\
Predefined configuration files for different dataset–model combinations are provided in the `configs/` directory.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config <config.yml>
```

## Inference

Specify the path to the trained model in the `ckpt` field of the configuration file.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config <config.yml>
```

## Checkpoints
Due to an issue with the Large model checkpoint, we release the Medium and Small checkpoints first.\
The checkpoint will be uploaded once the issue is resolved.

You can download the available model checkpoints from [here](https://drive.google.com/drive/folders/1qvPRNQ4KHpR409xMntJbBfEW1jnyKbV1?usp=sharing).


##  Citation
```bibtex
@article{lee2025moir,
  title={Moir$\backslash$'e Zero: An Efficient and High-Performance Neural Architecture for Moir$\backslash$'e Removal},
  author={Lee, Seungryong and Baek, Woojeong and Kim, Younghyun and Kim, Eunwoo and Moon, Haru and Yoo, Donggon and Park, Eunbyung},
  journal={arXiv preprint arXiv:2507.22407},
  year={2025}
}
```
