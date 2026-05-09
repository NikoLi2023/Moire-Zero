import argparse
import os
import time
import importlib

import cv2
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.utils as tvu

from models.mznet import MZNetLocal


def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def _strip_prefixes(key, prefixes):
    # Repeatedly strip known wrappers, e.g. module.mznet.xxx -> xxx
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def load_checkpoint_to_model(ckpt_path, model, device):
    # Load file (support .pth, .pth.tar, .tar)
    data = torch.load(ckpt_path, map_location=device)

    # Extract possible nested state dicts
    if isinstance(data, dict):
        if "state_dict" in data:
            state_dict = data["state_dict"]
        elif "model" in data and isinstance(data["model"], dict):
            state_dict = data["model"]
        else:
            state_dict = data
    else:
        state_dict = data

    # Normalize keys
    cleaned = {}
    for k, v in state_dict.items():
        new_k = _strip_prefixes(k, ("module.", "model.", "mznet."))
        cleaned[new_k] = v

    # Strict first, then fallback
    try:
        model.load_state_dict(cleaned, strict=True)
        print("Checkpoint loaded with strict=True")
    except RuntimeError as e:
        print("Strict load failed:", e)
        print("Retrying with strict=False (unmatched keys will be ignored).")
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print("Missing keys:", len(missing), "| Unexpected keys:", len(unexpected))


def _extract_pred(out):
    if isinstance(out, dict):
        return out.get("pred_x", list(out.values())[-1])
    if isinstance(out, (list, tuple)):
        # MZNetLocal returns (out2, out3, pred)
        if len(out) >= 3:
            return out[2]
        return out[-1]
    return out


def export_onnx(model, input_tensor, out_path, onnx_path=None):
    if importlib.util.find_spec("onnx") is None:
        print("ONNX package not installed, skipping ONNX export.")
        return

    if onnx_path is None:
        base, _ = os.path.splitext(out_path)
        onnx_path = base + ".onnx"

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            return _extract_pred(out)

    # Export on CPU for broader compatibility
    wrapper = ONNXWrapper(model).cpu().float().eval()
    example_input = input_tensor.cpu().float()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["pred"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "pred": {0: "batch", 2: "height", 3: "width"},
            },
            verbose=False,
        )
    print("ONNX saved:", onnx_path)


def infer_single(
    ckpt_path,
    image_path,
    out_path,
    config_filename="UHDM_s_1.yml",
    device=None,
    save_onnx=False,
    onnx_path=None,
    using_fp16=False,
):
    # Load config
    with open(os.path.join("configs", config_filename), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config = dict2namespace(cfg)

    # Device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    config.device = dev

    # Build model
    print("Creating model on:", dev)
    model = MZNetLocal(config)

    use_fp16 = bool(using_fp16 and dev.type == "cuda")
    if use_fp16:
        model = model.half()

    model.to(dev)
    model.eval()

    # Load checkpoint
    print("Loading checkpoint:", ckpt_path)
    load_checkpoint_to_model(ckpt_path, model, dev)

    # Load image
    img = Image.open(image_path).convert("RGB")
    t = transforms.ToTensor()(img).unsqueeze(0).to(dev)  # [1, 3, H, W]

    # For pure FP16 path, convert input dtype
    if use_fp16:
        t = t.half()

    # Pad to multiple of 32
    orig_h, orig_w = t.shape[2], t.shape[3]
    pad_h = (32 - (orig_h % 32)) % 32
    pad_w = (32 - (orig_w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

    # Forward
    start = time.time()
    with torch.no_grad():
        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(t)
        else:
            out = model(t)

    pred = _extract_pred(out)
    pred = torch.clamp(pred, 0.0, 1.0)
    pred = pred[:, :, :orig_h, :orig_w]
    elapsed = time.time() - start
    print(f"Inference elapsed: {elapsed:.3f} s")

    # Save image
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tvu.save_image(pred.float(), out_path)
    print("Saved:", out_path)

    # Optional ONNX export
    if save_onnx:
        try:
            export_onnx(model, t, out_path, onnx_path=onnx_path)
        except Exception as e:
            print("ONNX export failed:", e)


def build_parser():
    parser = argparse.ArgumentParser(description="Infer single image using checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth / .pth.tar / .tar)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--config", default="UHDM_s_1.yml", help="Config filename under configs/")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 inference (CUDA only)")
    parser.add_argument("--save-onnx", action="store_true", help="Export ONNX together")
    parser.add_argument("--onnx-path", default=None, help="Optional ONNX output path")
    return parser


if __name__ == "__main__":
    # args = build_parser().parse_args()

    total_start = time.time()
    # infer_single(
    #     ckpt_path=args.ckpt,
    #     image_path=args.image,
    #     out_path=args.out,
    #     config_filename=args.config,
    #     device=args.device,
    #     save_onnx=args.save_onnx,
    #     onnx_path=args.onnx_path,
    #     using_fp16=args.fp16,
    # )

    infer_single(
        ckpt_path=r"results/test_origin/ckpt/200.pth.tar",
        image_path=r"D:\Raw Data 8Bit(Remove Moire)\31WRF_S22U_WD518\oz_8bit.tif",
        out_path="results/_moire_demoire.tif",
        config_filename="UHDM_m_1.yml",
        device="cuda:0",
        save_onnx=False,
        onnx_path="",
        using_fp16=False,
    )

    src=cv2.imread("results/_moire_demoire.tif", cv2.IMREAD_COLOR)
    cv2.imwrite(r"D:\Raw Data 8Bit(Remove Moire)\31WRF_S22U_WD518\oz_8bit.tif", src)

    total_elapsed = time.time() - total_start
    print(f"Total elapsed: {total_elapsed:.3f} s")
