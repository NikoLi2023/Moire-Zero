import argparse
import os
import yaml
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as tvu
from models.mznet import MZNetLocal
import torch.nn.functional as F


def dict2namespace(config):
    import argparse as _arg
    ns = _arg.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_checkpoint_to_model(ckpt_path, model, device):
    # load file (support .pth, .pth.tar, .tar)
    data = torch.load(ckpt_path, map_location=device)
    # extract possible nested state dicts
    if isinstance(data, dict):
        if 'state_dict' in data:
            state_dict = data['state_dict']
        elif 'model' in data and isinstance(data['model'], dict):
            state_dict = data['model']
        else:
            state_dict = data
    else:
        state_dict = data

    # normalize keys by removing common prefixes
    def clean_state_dict(sd):
        new_state = {}
        for k, v in sd.items():
            new_k = k
            # remove typical prefixes
            for p in ('module.', 'mznet.', 'model.'):
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
            new_state[new_k] = v
        return new_state

    cleaned = clean_state_dict(state_dict)

    # try strict load first, fallback to non-strict to allow partial matches
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as e:
        print('Strict load failed:', e)
        print('Retrying load with strict=False (will ignore unmatched keys).')
        model.load_state_dict(cleaned, strict=False)


def infer_single(ckpt_path, image_path, out_path, config_filename='UHDM_m.yml', device=None):
    """Run inference for a single image using the given checkpoint.
    Only requires: checkpoint path, input image path, output image path.
    """
    # load config
    with open(os.path.join('configs', config_filename), 'r') as f:
        cfg = yaml.safe_load(f)
    config = dict2namespace(cfg)

    # device
    if device is None:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)
    config.device = dev

    # build model
    print('Creating model...')
    model = MZNetLocal(config)
    model.to(dev)
    model.eval()

    # load checkpoint
    print(f'Loading checkpoint: {ckpt_path}')
    load_checkpoint_to_model(ckpt_path, model, dev)
    print('Checkpoint loaded.')

    # load image
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    t = to_tensor(img).unsqueeze(0).to(dev)  # [1,3,H,W]

    # pad input to be multiple of 32 to avoid size mismatch in decoder
    orig_h, orig_w = t.shape[2], t.shape[3]
    pad_h = (32 - (orig_h % 32)) % 32
    pad_w = (32 - (orig_w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

    # forward
    with torch.no_grad():
        out = model(t)
        if isinstance(out, dict):
            if 'pred_x' in out:
                pred = out['pred_x']
            else:
                pred = list(out.values())[-1]
        elif isinstance(out, (list, tuple)):
            pred = out[2]
        else:
            raise RuntimeError('Unexpected model output type')

        pred = torch.clamp(pred, 0.0, 1.0)

    # crop back to original size if padded
    if 'orig_h' in locals() and (orig_h != pred.shape[2] or orig_w != pred.shape[3]):
        pred = pred[:, :, :orig_h, :orig_w]

    # save
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tvu.save_image(pred, out_path)
    print('Saved:', out_path)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Infer single image using checkpoint')
    # parser.add_argument('--ckpt', required=True, help='path to checkpoint (.pth or .pth.tar)')
    # parser.add_argument('--image', required=True, help='input moire image path')
    # parser.add_argument('--out', required=True, help='output image path')
    # parser.add_argument('--config', default='UHDM.yml', help='config filename in configs/ (default: UHDM.yml)')
    # parser.add_argument('--device', default=None, help='cuda or cpu (e.g. cuda:0)')
    # args = parser.parse_args()

    image_path = r"C:\Codes\Moire-Zero\Moire照片\0405_moire.jpg"
    out_path = r"C:\Codes\Moire-Zero\Moire照片\0405_moire_out.jpg"

    ckpt = r"C:\Codes\Moire-Zero\ckpt\MZNet_M_UHDM.pth"

    infer_single(ckpt, image_path, out_path)
