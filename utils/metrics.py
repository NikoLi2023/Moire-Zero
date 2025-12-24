import lpips
import torch
import numpy as np
from math import log10
from PIL import Image
import logging
import math
import os
import random
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from math import exp
from torchvision import transforms
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim



"""
A pytorch implementation for reproducing results in MATLAB, slightly modified from
https://github.com/mayorx/matlab_ssim_pytorch_implementation.
"""

import torch
import cv2
import numpy as np

def generate_1d_gaussian_kernel():
    return cv2.getGaussianKernel(11, 1.5)

def generate_2d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    return np.outer(kernel, kernel.transpose())

def generate_3d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    window = generate_2d_gaussian_kernel()
    return np.stack([window * k for k in kernel], axis=0)

class MATLAB_SSIM(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(MATLAB_SSIM, self).__init__()
        self.device = device
        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
        self.conv3d = conv3d.to(device)

        conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
        conv2d.weight.requires_grad = False
        conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
        self.conv2d = conv2d.to(device)

    def forward(self, img1, img2):
        assert len(img1.shape) == len(img2.shape)
        with torch.no_grad():
            img1 = torch.tensor(img1).to(self.device).float()
            img2 = torch.tensor(img2).to(self.device).float()

            if len(img1.shape) == 2:
                conv = self.conv2d
            elif len(img1.shape) == 3:
                conv = self.conv3d
            else:
                raise not NotImplementedError('only support 2d / 3d images.')
            return self._ssim(img1, img2, conv)

    def _ssim(self, img1, img2, conv):
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = conv(img1)
        mu2 = conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(img1 ** 2) - mu1_sq
        sigma2_sq = conv(img2 ** 2) - mu2_sq
        sigma12 = conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())

# ================================================================================================
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    
    
    return img_np.astype(out_type)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def pil_to_tensor(image, device):
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def calculate_mse(img1, img2):
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    mse = np.mean((img1_array - img2_array) ** 2)
    return mse


def save_combined_image(moire_image, result_image, gt_image, output_path):
    total_width = moire_image.width + result_image.width + gt_image.width
    max_height = max(moire_image.height, result_image.height)
    new_image = Image.new('RGB', (total_width, max_height))
    new_image.paste(moire_image, (0, 0))
    new_image.paste(result_image, (moire_image.width, 0))
    new_image.paste(gt_image, (moire_image.width + gt_image.width, 0))
    new_image.save(output_path)


def crop_image(image, patch_size=768, overlap=0):
    width, height = image.size
    patches = []
    for top in range(0, height, patch_size - overlap):
        for left in range(0, width, patch_size - overlap):
            right = min(left + patch_size, width)
            bottom = min(top + patch_size, height)
            patch = image.crop((left, top, right, bottom))
            patches.append(((left, top), patch))
    return patches


def reconstruct_image(patches, image_size):
    new_image = Image.new('RGB', image_size)
    for (left, top), patch in patches:
        new_image.paste(patch, (left, top))
    return new_image



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    """
    Fast pytorch implementation for SSIM, referred from
    "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py"
    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        

class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        psnr = -10*torch.log10(torch.mean((img1-img2)**2))
        
        return psnr
        

class create_metrics():
    """
       We note that for different benchmarks, previous works calculate metrics in different ways, which might
       lead to inconsistent SSIM results (and slightly different PSNR), and thus we follow their individual
       ways to compute metrics on each individual dataset for fair comparisons.
       For our 4K dataset, calculating metrics for 4k image is much time-consuming,
       thus we benchmark evaluations for all methods with a fast pytorch SSIM implementation referred from
       "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py".
    """
    def __init__(self, device, data_type="UHDM"):
        self.data_type = data_type
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.fast_ssim = SSIM()
        self.fast_psnr = PSNR()
        self.matlab_ssim = MATLAB_SSIM(device=device)
    


    def compute(self, out_img, gt):

        if self.data_type == 'UHDM':
            res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
        elif self.data_type == 'FHDMi':
            res_psnr, res_ssim = self.skimage_psnr_ssim(out_img, gt)
        elif self.data_type == 'TIP':
            res_psnr, res_ssim = self.matlab_psnr_ssim(out_img, gt)
        elif self.data_type == 'AIM':
            res_psnr, res_ssim = self.aim_psnr_ssim(out_img, gt)
        else:
            print('Unrecognized data_type for evaluation!')
            raise NotImplementedError

        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)

        # calculate LPIPS
        res_lpips = self.lpips_fn.forward(pre, tar, normalize=True).item()
        return res_lpips, res_psnr, res_ssim


    def fast_psnr_ssim(self, out_img, gt):
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)
        psnr = self.fast_psnr(pre, tar)
        ssim = self.fast_ssim(pre, tar)
        return psnr, ssim


    def skimage_psnr_ssim(self, out_img, gt):
        """
        Same with the previous SOTA FHDe2Net: https://github.com/PKU-IMRE/FHDe2Net/blob/main/test.py
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = ski_ssim(mt1, mi1, multichannel=True)
        return psnr, ssim

    def matlab_psnr_ssim(self, out_img, gt):
        """
        A pytorch implementation for reproducing SSIM results when using MATLAB
        same with the previous SOTA MopNet: https://github.com/PKU-IMRE/MopNet/blob/master/test_with_matlabcode.m
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = self.matlab_ssim(mt1, mi1)
        return psnr, ssim

    def aim_psnr_ssim(self, out_img, gt):
        """
        Same with the previous SOTA MBCNN: https://github.com/zhenngbolun/Learnbale_Bandpass_Filter/blob/master/main_multiscale.py
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        mi1 = mi1.astype(np.float32) / 255.0
        mt1 = mt1.astype(np.float32) / 255.0
        psnr = 10 * log10(1 / np.mean((mt1 - mi1) ** 2))
        
        ssim = ski_ssim(mt1, mi1, multichannel=True)
        return psnr, ssim 