"""
使用 ONNX 模型对单张图像进行推理并保存结果。
提供函数 infer_onnx_single(onnx_path, image_path, out_path, device='cpu', pad_to=32)
- onnx_path: ONNX 模型路径（由 infer_single.py 导出的文件）
- image_path: 输入图像路径（任意大小）
- out_path: 输出图像保存路径
- device: 'cpu' 或 'cuda'（默认 'cpu'；如果使用 GPU，需要安装 onnxruntime-gpu 并调整 provider）
- pad_to: 将输入 pad 到该整数的倍数（默认 32，和模型训练时 pad 保持一致）

简单的命令行用法也在 __main__ 中提供。
"""

import json
from PIL import Image
import numpy as np
import os
import time


def infer_onnx_single(
    onnx_path: str,
    image_path: str,
    out_path: str,
    device: str = "cpu",
    pad_to: int = 32,
    warmup_runs: int = 5,
    benchmark_runs: int = 30,
    enable_profile: bool = False,
    strict_cuda: bool = False,
):
    """
    使用 ONNX 模型对单张图像运行推理并保存输出图像。
    返回：输出图像的 numpy 数组（H, W, C，uint8）。

    参数说明：
    - warmup_runs: 预热次数，不计入平均耗时
    - benchmark_runs: 正式计时次数（仅统计 session.run）
    - enable_profile: 是否启用 ORT profiling 并输出 provider 摘要
    - strict_cuda: 请求 CUDA 且不可用时是否直接报错
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(
            "请先安装 onnxruntime (CPU: pip install onnxruntime) 才能使用此脚本。"
        ) from e

    available_providers = ort.get_available_providers()
    print("ORT available providers:", available_providers)

    # 选择 provider（并显式处理 CUDA 回退）
    want_cuda = bool(device and "cuda" in device.lower())
    if want_cuda and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif want_cuda:
        msg = (
            "Requested CUDA but CUDAExecutionProvider is unavailable. "
            "Falling back to CPUExecutionProvider."
        )
        if strict_cuda:
            raise RuntimeError(msg)
        print(msg)
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    if enable_profile:
        sess_options.enable_profiling = True

    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    print("ORT session providers (priority):", session.get_providers())

    # 读取图片并转换为 float32 [0,1], NCHW
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    h0, w0 = arr.shape[0], arr.shape[1]
    chw = arr.transpose(2, 0, 1)  # C,H,W
    input_np = np.expand_dims(chw, 0).astype(np.float32)  # 1,C,H,W

    # pad 到 pad_to 的倍数（使用 reflect，和模型推理时一致）
    pad_h = (pad_to - (h0 % pad_to)) % pad_to
    pad_w = (pad_to - (w0 % pad_to)) % pad_to
    if pad_h > 0 or pad_w > 0:
        pad = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))
        input_np = np.pad(input_np, pad, mode="reflect")

    input_name = session.get_inputs()[0].name

    # warmup
    for _ in range(max(0, warmup_runs)):
        session.run(None, {input_name: input_np})

    # benchmark（仅统计 session.run）
    run_lat_ms = []
    outputs = None
    for _ in range(max(1, benchmark_runs)):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: input_np})
        t1 = time.perf_counter()
        run_lat_ms.append((t1 - t0) * 1000.0)

    lat_np = np.array(run_lat_ms, dtype=np.float64)
    print(
        "ORT latency (session.run only) ms: "
        f"mean={lat_np.mean():.2f}, p50={np.percentile(lat_np, 50):.2f}, "
        f"p90={np.percentile(lat_np, 90):.2f}, min={lat_np.min():.2f}, max={lat_np.max():.2f}"
    )

    pred_np = outputs[0]  # 期望形状 1,C,H,W

    # crop 回原始尺寸（如果做了 pad）
    pred_np = pred_np[:, :, :h0, :w0]

    # 转为 uint8 图像并保存
    out_img = (
        (pred_np[0].transpose(1, 2, 0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    )
    out_pil = Image.fromarray(out_img)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_pil.save(out_path)

    # profiling 摘要（可用于判断是否存在 CPU 回退）
    if enable_profile:
        profile_file = session.end_profiling()
        print("ORT profile saved:", profile_file)
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            provider_count = {}
            for item in profile_data:
                provider = item.get("args", {}).get("provider")
                if provider:
                    provider_count[provider] = provider_count.get(provider, 0) + 1
            if provider_count:
                print("ORT profile provider usage:", provider_count)
            else:
                print("ORT profile provider usage: unavailable in profile entries.")
        except Exception as e:
            print("Failed to parse ORT profile JSON:", e)

    return out_img


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description='Use ONNX model to infer a single image')
    # parser.add_argument('--onnx', required=True, help='path to ONNX model')
    # parser.add_argument('--image', required=True, help='input image path')
    # parser.add_argument('--out', required=True, help='output image path')
    # parser.add_argument('--device', default='cpu', help="'cpu' or 'cuda' (default cpu)")
    # parser.add_argument('--pad', type=int, default=32, help='pad to multiple of this value (default 32)')
    # args = parser.parse_args()


    image_path = r"D:\Codes\UHDM\Moire\0405_moire.jpg"
    out_path = r"results\0405_moire_out_Onnx.png"

    onnx_file = r"results\_moire_out_FP32.onnx"

    # output_file = "C:/Codes/Moire-Zero/Moire照片/0405_moire_out_cuda.jpg"
    # record the elapsed time

    start_time = time.time()

    infer_onnx_single(
        onnx_file,
        image_path,
        out_path,
        device="cuda",
        warmup_runs=5,
        benchmark_runs=30,
        enable_profile=True,
        strict_cuda=False,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Whole Onnx Elapsed time: {elapsed_time:.2f} seconds")
