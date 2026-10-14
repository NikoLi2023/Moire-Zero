# -*- coding: utf-8 -*-
"""用法: python export_fp16.py <ckpt名> <配置名>  ->  ckpt/<名>_fp16.onnx"""
import os, sys, time, copy
os.chdir(r'C:\Codes\Moire-Zero'); sys.path.insert(0, r'C:\Codes\Moire-Zero')
import torch, yaml
import torch.nn.functional as F
from models.mznet import MZNetLocal, AvgPool2d

name, cfg_name = sys.argv[1], sys.argv[2]

def dict2namespace(c):
    import argparse as A
    ns = A.Namespace()
    for k, v in c.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

cfg = dict2namespace(yaml.safe_load(open(os.path.join('configs', cfg_name), encoding='utf-8')))
cfg.device = torch.device('cpu')
model = MZNetLocal(cfg).eval()
data = torch.load(f'ckpt/{name}.pth', map_location='cpu')
cleaned = {}
for k, v in data.items():
    nk = k
    for p in ('module.', 'mznet.', 'model.'):
        if nk.startswith(p): nk = nk[len(p):]
    cleaned[nk] = v
model.load_state_dict(cleaned, strict=True)

class _GlobalPool(torch.nn.Module):
    def forward(self, x): return F.adaptive_avg_pool2d(x, 1)
for parent in model.modules():
    for cname, child in parent.named_children():
        if isinstance(child, AvgPool2d):
            setattr(parent, cname, _GlobalPool())

class _W(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        out = self.m(x)
        return out.get('pred_x', list(out.values())[-1]) if isinstance(out, dict) else (out[2] if isinstance(out,(list,tuple)) else out)

half_model = _W(copy.deepcopy(model)).half().eval()
x = torch.rand(1, 3, 768, 1024, dtype=torch.float16)
t0 = time.time()
with torch.no_grad():
    torch.onnx.export(half_model, x, f'ckpt/{name}_fp16.onnx',
                      export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=['input'], output_names=['pred'],
                      dynamic_axes={'input': {0:'batch',2:'height',3:'width'},
                                    'pred': {0:'batch',2:'height',3:'width'}})
print(f'{name}_fp16.onnx 导出完成: {time.time()-t0:.1f}s '
      f'({os.path.getsize(f"ckpt/{name}_fp16.onnx")/1e6:.0f}MB)', flush=True)
