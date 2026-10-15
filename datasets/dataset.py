import os
import re
import glob
import random

import pandas as pd
import torch
import torch.utils.data
from PIL import Image

from datasets.data_augment import (
    PairCompose, 
    PairRandomCrop, 
    PairToTensor
)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def load_annotations(root_dir, csv_file, min_size=0):
    """读取 moiré/gt 配对表；CSV 不存在时自动扫描目录配对，免去手工制作 metadata.csv。

    返回与 pd.read_csv 相同结构的 DataFrame：第 0 列为 gt 相对路径，第 1 列为 moiré 相对路径。
    自动配对规则：
      - TIP 风格：存在 target/ 与 source/ 子目录时，按同名文件配对（相对 target/ 与 source/）；
      - UHDM / FHDMi 风格：同一目录内 <名字>_moire.<ext> 与 <名字>_gt.<ext> 配对。

    min_size > 0 时逐张检查分辨率，任一边小于 min_size 的配对不参训，
    并在启动阶段逐张打印提示（避免训练中途 RandomCrop 崩溃）。
    """
    pairs = []
    csv_ok = False
    csv_path = os.path.join(root_dir, csv_file)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # 校验表内文件真实存在；有缺失（如残留的示例表指向不存在文件）则回退自动扫描
        missing = 0
        for _, row in df.iterrows():
            if not os.path.isfile(os.path.join(root_dir, str(row.iloc[0]))) \
                    or not os.path.isfile(os.path.join(root_dir, str(row.iloc[1]))):
                missing += 1
        if missing == 0:
            pairs = [(str(row.iloc[0]), str(row.iloc[1])) for _, row in df.iterrows()]
            csv_ok = True
        else:
            print(f"=> 配对表 {csv_path} 中有 {missing}/{len(df)} 条记录的文件不存在，改用自动扫描配对")
    else:
        print(f"=> 未找到配对表 {csv_path}，自动扫描目录配对（无需手工制作 CSV）")

    if not csv_ok:
        tgt_dir = os.path.join(root_dir, 'target')
        src_dir = os.path.join(root_dir, 'source')
        if os.path.isdir(tgt_dir) and os.path.isdir(src_dir):
            # TIP 风格：target/ 与 source/ 同名配对
            for dirpath, _, files in os.walk(tgt_dir):
                for f in sorted(files):
                    if not f.lower().endswith(IMG_EXTS):
                        continue
                    rel = os.path.relpath(os.path.join(dirpath, f), tgt_dir)
                    if os.path.isfile(os.path.join(src_dir, rel)):
                        pairs.append((rel, rel))
                    else:
                        print(f"   [警告] 缺少与 target/{rel} 配对的 source 文件，已跳过")
        else:
            # UHDM / FHDMi 风格：目录内 <名字>_moire.<ext> ↔ <名字>_gt.<ext>
            for dirpath, _, files in os.walk(root_dir):
                for f in sorted(files):
                    low = f.lower()
                    if '_moire' not in low or not low.endswith(IMG_EXTS):
                        continue
                    stem = f[:low.index('_moire')]
                    gt_name = stem + '_gt' + os.path.splitext(f)[1]
                    gt_path = os.path.join(dirpath, gt_name)
                    if os.path.isfile(gt_path):
                        pairs.append((os.path.relpath(gt_path, root_dir),
                                      os.path.relpath(os.path.join(dirpath, f), root_dir)))
                    else:
                        print(f"   [警告] 未找到 {f} 对应的 {gt_name}，已跳过")

    if not pairs:
        raise FileNotFoundError(
            f"既未找到配对表 {csv_path}，也未能在 {root_dir} 下自动配对图像。"
            f"请按 <名字>_moire.<ext> / <名字>_gt.<ext> 命名，或手工提供 CSV。")

    if min_size > 0:
        pairs = _filter_small_images(pairs, root_dir, min_size)

    if not pairs:
        raise RuntimeError(f"{root_dir} 下没有满足分辨率 ≥{min_size}x{min_size} 的可训练图像。")
    print(f"=> 共 {len(pairs)} 对图像参与训练")
    return pd.DataFrame(pairs, columns=['gt', 'moire'])


def _filter_small_images(pairs, root_dir, min_size):
    """剔除分辨率低于 min_size 的配对，逐张打印被跳过的照片（PIL 只读文件头，速度快）。"""
    kept, skipped = [], []
    for gt_rel, moire_rel in pairs:
        size_str = None
        for rel in (moire_rel, gt_rel):
            try:
                with Image.open(os.path.join(root_dir, rel)) as im:
                    w, h = im.size
            except Exception as e:
                size_str = f"无法读取（{type(e).__name__}）"
                break
            if w < min_size or h < min_size:
                size_str = f"{w}x{h}"
                break
        if size_str is None:
            kept.append((gt_rel, moire_rel))
        else:
            skipped.append((moire_rel, size_str))
    if skipped:
        print(f"=> [提示] 以下 {len(skipped)} 张照片分辨率低于 {min_size}x{min_size}，不参训：")
        for name, info in skipped:
            print(f"   {name}（{info}）")
    return kept
    
    
class UHDMdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = UHDMMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                          mode='train',
                                          patch_size=self.config.data.patch_size,
                                          csv_file=self.config.data.train_data_csv)
        
        val_dataset = UHDMMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'test'),
                                        mode='test',
                                        patch_size=self.config.data.patch_size,
                                        csv_file=self.config.data.val_data_csv)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
  

class UHDMMoireDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, csv_file, mode='train', patch_size=256):
        # 训练模式下过滤小于 patch_size 的图（验证整图进网络，不做限制）
        self.annotations = load_annotations(root_dir, csv_file,
                                            min_size=patch_size if mode == 'train' else 0)
        self.root_dir = root_dir 
        self.num_images = len(self.annotations) 
        self.mode = mode
        self.patch_size = patch_size
        if self.mode=='train':
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])
            
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gt_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 0])
        moire_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 1])
        gt_image = Image.open(gt_name).convert('RGB')
        moire_image = Image.open(moire_name).convert('RGB')
        moire_image, gt_image = self.transforms(moire_image, gt_image)
        image_id = self.annotations.iloc[idx, 0][:4]
        return torch.cat([moire_image, gt_image], dim=0), image_id


class FHDMIdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = FHDMIMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                          mode='train',
                                          patch_size=self.config.data.patch_size,
                                          csv_file=self.config.data.train_data_csv)
        
        val_dataset = FHDMIMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'test'),
                                        mode='test',
                                        patch_size=self.config.data.patch_size,
                                        csv_file=self.config.data.val_data_csv)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class FHDMIMoireDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, csv_file, mode='train', patch_size=256):
        # 训练模式下过滤小于 patch_size 的图（验证整图进网络，不做限制）
        self.annotations = load_annotations(root_dir, csv_file,
                                            min_size=patch_size if mode == 'train' else 0)
        self.root_dir = root_dir
        self.num_images = len(self.annotations) 
        self.mode = mode
        self.patch_size = patch_size
        if self.mode=='train':
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])
            
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gt_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 0])
        moire_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 1])
        gt_image = Image.open(gt_name).convert('RGB')
        moire_image = Image.open(moire_name).convert('RGB')
        moire_image, gt_image = self.transforms(moire_image, gt_image)
        image_id = self.annotations.iloc[idx, 0][-9:-4]
        return torch.cat([moire_image, gt_image], dim=0), image_id
    
    
class TIPdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        
        train_dataset = TIPMoireDataset(root_dir=os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                        csv_file=self.config.data.train_data_csv,
                                        mode='train',
                                        patch_size=self.config.data.patch_size,
                                        )
        
        val_dataset = TIPMoireDataset(root_dir=os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'test'),
                                        csv_file=self.config.data.val_data_csv,
                                        mode='test',
                                        patch_size=self.config.data.patch_size,
                                        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
    

class TIPMoireDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, csv_file, mode='train', patch_size=256):
        self.annotations = load_annotations(root_dir, csv_file)
        self.root_dir = root_dir
        self.num_images = len(self.annotations) 
        self.mode = mode
        self.patch_size = patch_size
        self.transforms = PairCompose([
                PairToTensor()
            ])
            
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gt_name = os.path.join(os.path.join(self.root_dir, 'target'), self.annotations.iloc[idx, 0])
        moire_name = os.path.join(os.path.join(self.root_dir, 'source'), self.annotations.iloc[idx, 1])
        gt_image = Image.open(gt_name).convert('RGB')
        moire_image = Image.open(moire_name).convert('RGB')
        w, h = gt_image.size
        
        if self.mode == 'train':
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            gt_image = gt_image.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            moire_image = moire_image.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            gt_image = gt_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            moire_image = moire_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            
        elif self.mode == 'test':
            gt_image = gt_image.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            moire_image = moire_image.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            gt_image = gt_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            moire_image = moire_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        moire_image, gt_image = self.transforms(moire_image, gt_image)
        image_id = self.annotations.iloc[idx, 0][:-11]
        return torch.cat([moire_image, gt_image], dim=0), image_id
