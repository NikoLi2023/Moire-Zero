import argparse
import os
import time
import importlib
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

    if onnx_path is None or os.path.isdir(onnx_path):
        base, _ = os.path.splitext(out_path)
        onnx_path = base + ".onnx"

    # 路径语义: 目标以 _fp16.onnx 结尾 -> 只导出 fp16 到该路径;
    # 否则 -> 导出 fp32 到该路径, 并自动派生 <stem>_fp16.onnx 孪生。
    fp16_only = os.path.basename(onnx_path).endswith("_fp16.onnx")

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            return _extract_pred(out)

    # 关键: SCA 里的自定义 AvgPool2d(NAFNet Local 近似)的 kernel 窗口在 tracing 时
    # 会被固化为常量, 导致导出图只认导出时的尺寸(H/W<=1536 直接崩溃)。
    # 替换为真正的全局平均池化后: 1) 无 TracerWarning; 2) 任意尺寸泛化;
    # 3) 不再有 edge/replicate 模式的 Pad, 兼容 CUDA EP / TensorRT。
    from models.mznet import AvgPool2d

    class _GlobalPool(torch.nn.Module):
        def forward(self, x):
            return F.adaptive_avg_pool2d(x, 1)

    n_replaced = 0
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, AvgPool2d):
                setattr(parent, name, _GlobalPool())
                n_replaced += 1
    print(f"ONNX export: replaced {n_replaced} custom AvgPool2d with global avg pool.")

    common_kwargs = dict(
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

    example_input = input_tensor.cpu().float()

    if fp16_only:
        # 只导出 fp16 到指定的 *_fp16.onnx 路径(配合 using_fp16=True 的 GPU 工作流)
        half_wrapper = ONNXWrapper(model).cpu().half().eval()
        with torch.no_grad():
            torch.onnx.export(
                half_wrapper, example_input.half(), onnx_path, **common_kwargs
            )
        print("FP16 ONNX saved:", onnx_path)
        return

    # Export fp32 on CPU for broader compatibility
    wrapper = ONNXWrapper(model).cpu().float().eval()

    with torch.no_grad():
        torch.onnx.export(wrapper, example_input, onnx_path, **common_kwargs)
    print("ONNX saved:", onnx_path)

    # 同时导出 fp16 版本(<stem>_fp16.onnx), 供 GPU 推理使用:
    # fp32 全尺寸激活值(单张 concat 可达 5.5GB)放不进 12GB 显存,
    # fp16 显存减半且 GPU 走 tensor core 更快, 精度损失肉眼不可感知(实测 mean|diff| ~ 5e-4)。
    # infer_onnx_single.py 在 device='cuda' 时会自动优先加载 *_fp16.onnx。
    import copy as _copy

    fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
    half_wrapper = _copy.deepcopy(wrapper).half().eval()
    with torch.no_grad():
        torch.onnx.export(half_wrapper, example_input.half(), fp16_path, **common_kwargs)
    print("FP16 ONNX saved:", fp16_path)


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
        # t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant",value=0.0)

    #创建一个Tdummy其数据结构和t一样，只是数值随机
    Tdummy = torch.randn_like(t)

    startTime = time.time()
    # forward
    with torch.no_grad(), torch.autocast(device_type='cuda',
                                         dtype=torch.float16):  # 开启一个上下文，在该上下文内 PyTorch 不会计算梯度或记录计算图，用于推理以节省显存和计算时间。
        out = model(Tdummy)  # 将输入张量 t 传入模型前向计算，得到模型输出 out。输出类型取决于模型实现（可能是 tuple/list/dict）。
        if isinstance(out, dict):
            if 'pred_x' in out:
                pred = out['pred_x']
            else:
                pred = list(out.values())[-1]
        elif isinstance(out, (list, tuple)):
            pred = out[2]
        else:
            raise RuntimeError('Unexpected model output type')

        pred = torch.clamp(pred, 0.0, 1.0)  # 把模型输出张量的像素值限制在合法的图像范围 [0.0, 1.0]，以便后续保存/可视化且避免超出范围的数值。

    end_time = time.time()
    elapsed_time = end_time - startTime
    print(f"Inference Elapsed time1: {elapsed_time:.2f} seconds")
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
    print(f"Inference elapsed2: {elapsed:.3f} s")

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
    print(f"Inference elapsed3: {elapsed:.3f} s")

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
    total_start = time.time()
    # imgList=[img for img in os.listdir(r"C:\Users\1\Desktop\origin") if img.endswith(".tif")]
    # renameList = [f"{os.path.splitext(i)[0]}_Demoire.tif" for i in imgList]
    # for i in range(len(imgList)):
    #     img = imgList[i]
    #     out = renameList[i]
    #     infer_single(
    #         ckpt_path=r"ckpt/MZNet_S_UHDM.pth",
    #         image_path=rf"C:\Users\1\Desktop\origin\{img}",
    #         out_path=fr"C:\Users\1\Desktop\origin/{out}",
    #         config_filename="UHDM_m_1.yml",
    #         device="cuda:0",
    #         save_onnx=True,
    #         onnx_path="results",
    #         using_fp16=False,
    #     )

    infer_single(
        ckpt_path=r"ckpt/MZNet_S_UHDM.pth",
        image_path=rf"Moire照片/0405_moire.jpg",
        out_path=fr"Moire照片/0405_moire_out.jpg",
        config_filename="UHDM_s.yml",
        device="cuda:0",
        save_onnx=True,
        onnx_path="ckpt/MZNet_S_UHDM_fp16.onnx",
        using_fp16=True,
    )
    total_elapsed = time.time() - total_start
    print(f"Total elapsed: {total_elapsed:.3f} s")
