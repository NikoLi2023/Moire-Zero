import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)


class PairRandomCropNumpy:
    """numpy(HWC) 图像的成对随机裁剪，语义同 torchvision RandomCrop（图小于裁剪尺寸时边缘补边）。"""

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, label):
        th, tw = self.size
        h, w = image.shape[:2]
        if h < th or w < tw:
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            label = np.pad(label, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            h, w = image.shape[:2]
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return image[i:i + th, j:j + tw], label[i:i + th, j:j + tw]


class PairToTensorNumpy:
    """numpy(HWC, uint8/uint16/float) -> CHW float32 [0,1]，按位深归一化：
    uint8 除以 255，uint16 除以 65535，浮点直接裁剪到 [0,1]。
    与训练网络的 [0,1] 输入空间约定保持一致，16-bit 照片信息无损进入训练。"""

    def __call__(self, pic, label):
        return self._to_tensor(pic), self._to_tensor(label)

    @staticmethod
    def _to_tensor(a):
        a = np.ascontiguousarray(a)
        if a.dtype == np.uint8:
            scale = 255.0
        elif a.dtype == np.uint16:
            scale = 65535.0
        else:
            return torch.from_numpy(a.transpose(2, 0, 1)).float().clamp_(0.0, 1.0)
        return torch.from_numpy(a.transpose(2, 0, 1)).float().div_(scale)
