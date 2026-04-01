import os
import csv
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from utils import logging, optimize
from utils.loss_utill import multi_VGGPerceptualLoss
from utils.metrics import *
from models.mznet import MZNetLocal



def to_float(val):
    return val.item() if isinstance(val, torch.Tensor) else val


def format_eta(seconds):
    days = seconds // (24 * 3600)
    hours = (seconds % (24 * 3600)) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)}d {int(hours):02}:{int(minutes):02}:{int(secs):02}"
    else:
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"
    

class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device
        self.mznet = MZNetLocal(config)

    def forward(self, x):
        data_dict = {}
        input_img = x[:, :3, :, :]
        out2, out3, pred_x = self.mznet(input_img)    
        data_dict["pred_x"] = pred_x
        data_dict["out2"] = out2
        data_dict["out3"] = out3
        return data_dict
    

class Pipeline(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)
        self.loss_fn = multi_VGGPerceptualLoss(lam=1, lam_p=1).to(self.device)
        self.optimizer, self.scheduler = optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        
        image_folder = os.path.join(self.config.training.result_folder, self.config.data.train_dataset)
        os.makedirs(image_folder, exist_ok=True)

    def load_ddm_ckpt(self, load_path):
       
        print(f"=> Loading checkpoint from {load_path}")
        def clean_state_dict(state_dict):
            """Remove 'module.' prefix if present"""
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_k] = v
            return new_state_dict

        if load_path.endswith(".tar"):
            checkpoint = utils.logging.load_checkpoint(load_path, None)
            state_dict = checkpoint['state_dict']
            # Clean if necessary
            if any(k.startswith('module.') for k in state_dict.keys()):
                print("=> Detected 'module.' prefix, removing it.")
                state_dict = clean_state_dict(state_dict)
            
            self.model.load_state_dict(state_dict, strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            self.step = checkpoint['step']
            self.start_epoch = checkpoint['epoch']

            print(f"=> Loaded full checkpoint {load_path} (step {self.step})")
        
        elif load_path.endswith(".pth"):
            state_dict = torch.load(load_path)
            if any(k.startswith('module.') for k in state_dict.keys()):
                print("=> Detected 'module.' prefix, removing it.")
                state_dict = clean_state_dict(state_dict)

            self.model.load_state_dict(state_dict, strict=True)
            print(f"=> Loaded state_dict only from {load_path}")


    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.config.training.resume):
            self.load_ddm_ckpt(self.config.training.resume)
        else:
            print("=> No checkpoint found, starting training from scratch.")
        
        max_grad_norm = 5
        total_steps = (self.config.training.n_epochs - self.start_epoch) * len(train_loader)
        completed_steps = 0
        batch_times = []
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            total_loss = 0
            self.optimizer.zero_grad()
            with tqdm(train_loader, unit="batch") as tepoch:
                for i, (x, y) in enumerate(tepoch):
                    start_batch = time.time()

                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                    self.model.train()

                    x = x.to(self.device)
                    output = self.model(x)
                    loss = self.estimation_loss(output, x[:, 3:, :, :])
                    loss = loss / self.config.training.accumulation_steps
                    loss.backward()
                    
                    total_loss += loss.item()
                    avg_train_loss = total_loss / (i+1)
                    
                    batch_time = time.time() - start_batch
                    batch_times.append(batch_time)
                    completed_steps += 1
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    remaining_steps = total_steps - completed_steps
                    eta_seconds = avg_batch_time * remaining_steps
                    eta_formatted = format_eta(eta_seconds)
                    desc="Epoch {}: step:{}, lr:{:.6f}, loss:{:.4f}".format(epoch, self.step, self.scheduler.get_last_lr()[0], avg_train_loss)
                    tepoch.set_description(desc)
                    tepoch.set_postfix(ETA_total=eta_formatted)
                    tepoch.update()
                    
                    if (i+1) % self.config.training.accumulation_steps == 0  or (i + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)           
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.step += 1
                        if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                            self.model.eval()
                            self.sample_validation_patches(val_loader, self.step)

                            utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                        'state_dict': self.model.state_dict(),
                                                        'optimizer': self.optimizer.state_dict(),
                                                        'scheduler': self.scheduler.state_dict(),
                                                        'params': self.args,
                                                        'config': self.config},
                                                        filename=os.path.join(self.config.training.result_folder, self.config.data.train_dataset, 'ckpt', f'{self.step}'))
                            
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                    'state_dict': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler.state_dict(),
                                    'params': self.args,
                                    'config': self.config},
                                    filename=os.path.join(self.config.training.result_folder, self.config.data.train_dataset, 'ckpt', f'{self.step}'))



    def estimation_loss(self, output, target):
        
        out2, out3, pred =output["out2"], output["out3"], output["pred_x"]
        loss = self.loss_fn(pred, out3, out2, target) 
        return loss
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.config.training.result_folder, self.config.data.train_dataset)
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):

                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), f"{y[0]}.png"))
            del x, out, pred_x
            torch.cuda.empty_cache()
            
    def eval(self, val_loader):
        if os.path.isfile(self.config.eval.ckpt):
            self.load_ddm_ckpt(self.config.eval.ckpt)
            self.model.eval()
        else:
            print('Pre-trained model path is missing!')
            
        image_folder = os.path.join(self.config.eval.result_folder, self.config.data.val_dataset)
        metrics = create_metrics(self.device, self.config.data.data_type)
        metrics_data = []  
        lpips_vals = []
        psnr_vals = []
        ssim_vals = []

        tbar = tqdm(val_loader)
        with torch.no_grad():
                                        
            for i, (x, y) in enumerate(tbar):
                x_cond = x[:, :3, :, :].to(self.device)
                gt = x[:, 3:, :, :].to(self.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.model(x_cond)["pred_x"]
                x_output = x_output[:, :, :h, :w]
                x_output = torch.clip(x_output, min=0, max=1)
                lpips_val, psnr_val, ssim_val = metrics.compute(x_output, gt)
      
                lpips_vals.append(lpips_val)
                psnr_vals.append(to_float(psnr_val))
                ssim_vals.append(to_float(ssim_val))
                metrics_data.append([y, lpips_val, to_float(psnr_val), to_float(ssim_val)])
                tbar.set_description(
                    f"Avg. LPIPS: {sum(lpips_vals) / len(lpips_vals):.4f}, "
                    f"Avg. PSNR: {sum(psnr_vals) / len(psnr_vals):.4f}, "
                    f"Avg. SSIM: {sum(ssim_vals) / len(ssim_vals):.4f}"
                )
                
                if self.config.eval.save_img:
                    utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
            
                del x, y, x_cond, gt, x_output
                torch.cuda.empty_cache()
            
            mean_lpips = sum(lpips_vals) / len(lpips_vals)
            mean_psnr = sum(psnr_vals) / len(psnr_vals)
            mean_ssim = sum(ssim_vals) / len(ssim_vals)
            print(f"Mean LPIPS: {mean_lpips}, Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}")
            
            metrics_data.append(["Mean", mean_lpips, mean_psnr, mean_ssim])
        
        with open(os.path.join(self.config.eval.result_folder, f"{self.config.data.val_dataset}_{self.config.eval.ckpt.split('/')[-1][:-8]}.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "LPIPS", "PSNR", "SSIM"])
            writer.writerows(metrics_data)
