"""MZNet 快捷训练入口：命令行参数直接覆盖 yml，无需手工编辑配置文件。

用法示例（在仓库根目录运行）：

  # 最小用法：只指定数据父目录，其余沿用 configs/UHDM_s_1.yml
  python quick_train.py --data_dir D:/datasets

  # data_dir 是父目录，后面会自动拼 <data_dir>/<train_dataset>/train
python quick_train.py --data_dir D:/mydata --train_dataset uhdm_train

  # 从官方预训练权重继续训练（.pth = 仅加载权重，优化器从头开始） 
  python quick_train.py --data_dir D:/datasets --resume ckpt/MZNet_S_UHDM.pth --epochs 300

  # 断点续训（.pth.tar = 完整 checkpoint，含优化器/epoch，从上次接着跑）
  python quick_train.py --data_dir D:/datasets --resume results/xxx/checkpoint.pth.tar

  # 换基础配置 + 调批量/学习率/任务名
  python quick_train.py --config UHDM_m_1.yml --data_dir D:/datasets --batch_size 1 --lr 0.0004 --name my_run

  # 训练结束权重自定义命名，并自动导出真 fp16 ONNX（results/<任务名>/MZNet_S_UHDM_ft.onnx）
  python quick_train.py --data_dir . --train_dataset test_origin --val_dataset test_origin ^
      --resume ckpt/MZNet_S_UHDM.pth --ckpt_name MZNet_S_UHDM_ft --export_onnx

说明：
  - 配对表 CSV 不再是必需的：数据集目录里能按 <名字>_moire.<ext> / <名字>_gt.<ext>
    （或 TIP 的 source/target 同名）自动配对即可；metadata.csv 存在且有效时仍优先按表读取。
  - --resume 传 .pth 只加载模型权重；传训练中保存的 .pth.tar 才是严格意义的续训。
  - --ckpt_name 指定训练结束时最终权重文件名（不含扩展名，自动补 .pth.tar；
    中间过程的 checkpoint 仍按 {step}.pth.tar 命名）。
  - --export_onnx 在训练结束后直接从内存模型导出真 fp16 ONNX（权重+输入输出均 float16），
    C# 侧 UHDMSOnnxRunner 可直接加载；默认输出到结果目录、文件名为"权重名+.onnx"
    （即 <result_folder>/<ckpt_name>.onnx）；--onnx_path 可显式指定其他输出路径。
"""
import argparse
import os

import numpy as np
import torch
import yaml

import datasets
from models import Pipeline


def parse_args():
    p = argparse.ArgumentParser(description='快捷训练（命令行覆盖 yml）')
    p.add_argument('--config', default='GQMoire.yml', help='configs/ 下的基础配置文件名')
    # 数据
    p.add_argument('--data_dir', default=None, help='数据集父目录，即 <data_dir>/<数据集名>/train|test')
    p.add_argument('--train_dataset', default=None, help='训练数据集目录名（默认沿用基础配置）')
    p.add_argument('--val_dataset', default=None, help='验证数据集目录名')
    p.add_argument('--train_data_csv', default=None, help='配对表文件名（不配则自动扫描配对）')
    p.add_argument('--val_data_csv', default=None)
    p.add_argument('--patch_size', type=int, default=None)
    # 训练
    p.add_argument('--resume', default=None, help='.pth=仅权重；.pth.tar=完整续训')
    p.add_argument('--name', default=None, help='训练任务名（结果保存目录）')
    p.add_argument('--result_folder', default=None, help='训练结果保存路径')
    p.add_argument('--epochs', type=int, default=None, help='总 epoch 数')
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    # 保存与导出
    p.add_argument('--ckpt_name', default=None,
                   help='训练结束时权重文件名（不含扩展名，自动补 .pth.tar；默认 {step}.pth.tar）')
    p.add_argument('--export_onnx', action='store_true',
                   help='训练结束后自动导出真 fp16 ONNX（复用 infer_single.export_onnx）')
    p.add_argument('--onnx_path', default=None,
                   help='ONNX 输出路径（默认 <结果目录>/<ckpt_name>.onnx，即"权重名+.onnx"）')
    return p.parse_args()


def main():
    args = parse_args()

    with open(os.path.join('configs', args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 命令行参数覆盖基础配置（None 表示不覆盖）
    overrides = [
        ('data', 'data_dir', args.data_dir),
        ('data', 'train_dataset', args.train_dataset),
        ('data', 'val_dataset', args.val_dataset),
        ('data', 'train_data_csv', args.train_data_csv),
        ('data', 'val_data_csv', args.val_data_csv),
        ('data', 'patch_size', args.patch_size),
        ('training', 'resume', args.resume),
        ('training', 'name', args.name),
        ('training', 'result_folder', args.result_folder),
        ('training', 'n_epochs', args.epochs),
        ('training', 'batch_size', args.batch_size),
        ('training', 'ckpt_name', args.ckpt_name),
        ('optim', 'lr', args.lr),
    ]
    for section, key, value in overrides:
        if value is not None:
            cfg[section][key] = value
            print(f"=> 覆盖配置 {section}.{key} = {value}")

    # result_folder 为空时给默认值，避免 pipeline 拼接路径报错
    if not cfg['training'].get('result_folder'):
        cfg['training']['result_folder'] = os.path.join('results', cfg['training']['name'])
        print(f"=> 覆盖配置 training.result_folder = {cfg['training']['result_folder']}")

    import train as train_module
    config = train_module.dict2namespace(cfg)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: {}'.format(device))
    config.device = device

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    print("=> Using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)

    print('=> Creating model...')
    pipeline = Pipeline(args, config)
    print(sum(p.numel() for p in pipeline.model.parameters()))
    final_ckpt = pipeline.train(DATASET)
    print('=> 训练完成，最终权重：', final_ckpt)

    if args.export_onnx:
        export_fp16_onnx(pipeline, args.onnx_path,
                         args.ckpt_name or cfg['training']['name'],
                         cfg['training']['result_folder'])


def export_fp16_onnx(pipeline, onnx_path, base_name, result_folder):
    """训练结束后直接从内存中的模型导出真 fp16 ONNX（与 using_fp16=True 工作流一致）。

    默认输出到训练结果目录：<result_folder>/<base_name>.onnx，
    即最终文件名为"权重名.onnx"（base_name 即 --ckpt_name）。
    """
    if onnx_path is None:
        os.makedirs(result_folder, exist_ok=True)
        onnx_path = os.path.join(result_folder, f'{base_name}.onnx')
    else:
        out_dir = os.path.dirname(os.path.abspath(onnx_path))
        os.makedirs(out_dir, exist_ok=True)

    # export_onnx 的路径语义：文件名以 _fp16.onnx 结尾 -> 只导出真 fp16 单文件。
    # 先在目标目录导到临时 _fp16 名保证走 fp16 单文件分支，再改名为最终文件名。
    root, _ = os.path.splitext(onnx_path)
    tmp_fp16 = root + '_fp16.onnx'

    from infer_single import export_onnx
    pipeline.model.eval()
    dummy = torch.randn(1, 3, 256, 256)  # 导出用 dummy 输入，动态尺寸由 dynamic_axes 保证
    print('=> 训练后自动导出 fp16 ONNX：', onnx_path)
    export_onnx(pipeline.model, dummy, tmp_fp16, onnx_path=tmp_fp16)
    if os.path.abspath(tmp_fp16) != os.path.abspath(onnx_path):
        os.replace(tmp_fp16, onnx_path)


if __name__ == '__main__':
    print('Starting training (quick_train)...')
    main()
    print('Training finished!')
