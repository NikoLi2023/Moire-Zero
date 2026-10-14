"""
使用 ONNX 模型对单张图像进行推理并保存结果。
提供函数 infer_onnx_single(onnx_path, image_path, out_path, device='cpu', pad_to=32)
- onnx_path: ONNX 模型路径（由 infer_single.py 导出的文件）
- image_path: 输入图像路径（任意大小）
- out_path: 输出图像保存路径
- device: 'cpu' 或 'cuda'（默认 'cpu'）
- pad_to: 将输入 pad 到该整数的倍数（默认 32，和模型训练时 pad 保持一致）

GPU 说明（重要）:
1. 本机 onnxruntime-gpu 的 CUDA 依赖(cuDNN 9 / CUDA 12 DLL)不在系统 PATH 上,
   脚本会在请求 CUDA 时自动把 PyTorch 自带的 DLL 目录加入 PATH 后再加载 provider;
2. GPU 上若给定 fp32 模型且同目录存在 *_fp16.onnx, 会自动换用 fp16 模型
   (fp32 全尺寸激活值单张 concat 可达 5.5GB, 放不进 12GB 显存);
   直接传入 *_fp16.onnx 也能正确识别(以模型实际输入类型为准);
3. CUDA provider 显式配置: HEURISTIC 卷积搜索(避免新尺寸首次推理数十秒)、
   显存上限 11GB、kSameAsRequested 分配策略(避免 arena 反复扩缩导致的数量级劣化)。

简单的命令行用法也在 __main__ 中提供。
"""

from PIL import Image
import numpy as np
import os
import time


def _inject_torch_cuda_dlls():
    """把 torch 自带的 CUDA/cuDNN DLL 目录加到 PATH(需在创建 InferenceSession 前调用)"""
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib) and lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _onnx_input_is_fp16(path):
    """读取 ONNX 模型输入类型, 判断是否为真 fp16(防止名为 *_fp16 实为 fp32 的坏文件)"""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        return m.graph.input[0].type.tensor_type.elem_type == 10  # 10 = FLOAT16
    except Exception:
        return None  # 无法判断时不阻止


def infer_onnx_single(
    onnx_path: str,
    image_path: str,
    out_path: str,
    device: str = "cpu",
    pad_to: int = 32,
):
    """
    使用 ONNX 模型对单张图像运行推理并保存输出图像。
    返回：输出图像的 numpy 数组（H, W, C，uint8）。
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(
            "请先安装 onnxruntime (CPU: pip install onnxruntime) 才能使用此脚本。"
        ) from e

    use_cuda = bool(device and "cuda" in device.lower())

    # GPU: 注入 torch 的 CUDA DLL; 若给定的是 fp32 模型且同目录存在 fp16 版本, 自动换用(显存减半)。
    # 注意: 是否 fp16 一律以模型实际输入类型为准(见下方 session 创建后的检测),
    #       直接传入 *_fp16.onnx 也能正确识别。
    model_path = onnx_path
    if use_cuda:
        _inject_torch_cuda_dlls()
        root, ext = os.path.splitext(onnx_path)
        if root.endswith("_fp16"):
            print(f"GPU 模式: 使用 fp16 模型 {os.path.basename(onnx_path)}")
        else:
            candidate = root + "_fp16" + ext
            if os.path.exists(candidate) and _onnx_input_is_fp16(candidate):
                model_path = candidate
                print(f"GPU 模式: 使用 fp16 模型 {os.path.basename(candidate)}")
            else:
                if os.path.exists(candidate):
                    print(f"[警告] {os.path.basename(candidate)} 不是真 fp16 模型(输入仍为 fp32), "
                          f"已忽略。请用 export_fp16.py 重新导出。")
                else:
                    print(f"[提示] 未找到 fp16 模型 {os.path.basename(candidate)}, "
                          f"fp32 全尺寸大图可能因显存不足而极慢。")

    providers = ["CPUExecutionProvider"]
    provider_options = None
    if use_cuda:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{
                "device_id": 0,
                "cudnn_conv_algo_search": "HEURISTIC",   # 避免新尺寸首次推理的穷举搜索
                "gpu_mem_limit": 11 * 1024**3,           # 显存上限(4070 Ti 12GB)
                "arena_extend_strategy": "kSameAsRequested",  # 避免 arena 反复扩缩
            }, {}]
        else:
            print(f"[警告] 请求了 CUDA, 但不可用 (available: {avail}), 已回退到 CPU。")

    t_create = time.time()
    if provider_options is not None:
        session = ort.InferenceSession(
            model_path, providers=providers, provider_options=provider_options
        )
    else:
        session = ort.InferenceSession(model_path, providers=providers)
    print(f"Session 创建(一次性开销): {time.time() - t_create:.2f}s, "
          f"实际 providers: {session.get_providers()}")

    # 是否 fp16 以模型实际输入类型为准(直接传入 *_fp16.onnx 也能正确识别)
    fp16 = "float16" in (session.get_inputs()[0].type or "")

    # 读取图片并转换为 float [0,1], NCHW
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    h0, w0 = arr.shape[0], arr.shape[1]
    chw = arr.transpose(2, 0, 1)  # C,H,W
    input_np = np.expand_dims(chw, 0)  # 1,C,H,W

    # pad 到 pad_to 的倍数（使用 reflect，和模型推理时一致）
    pad_h = (pad_to - (h0 % pad_to)) % pad_to
    pad_w = (pad_to - (w0 % pad_to)) % pad_to
    if pad_h > 0 or pad_w > 0:
        input_np = np.pad(input_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                          mode="reflect")

    input_np = input_np.astype(np.float16 if fp16 else np.float32)

    # 预热: 用小尺寸 dummy 输入初始化线程池/分配器/cuDNN 句柄, 避免全尺寸预热浪费时间
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, 64, 64), dtype=input_np.dtype)
    session.run(None, {input_name: dummy})

    t_infer = time.time()
    outputs = session.run(None, {input_name: input_np})
    print(f"纯推理时间 (session.run): {time.time() - t_infer:.2f}s")
    pred_np = outputs[0].astype(np.float32)  # 期望形状 1,C,H,W

    # crop 回原始尺寸（如果做了 pad）
    pred_np = pred_np[:, :, :h0, :w0]

    # 转为 uint8 图像并保存
    out_img = (
        (pred_np[0].transpose(1, 2, 0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    )
    out_pil = Image.fromarray(out_img)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_pil.save(out_path)

    return out_img


if __name__ == "__main__":
    onnx_file = "ckpt/MZNet_S_UHDM_fp16.onnx"
    image_file = "Moire照片/0405_moire.jpg"
    output_file = "Moire照片/0405_moire_out_cuda.jpg"
    device = "cuda:0"
    start_time = time.time()
    infer_onnx_single(onnx_file, image_file, output_file,device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Whole Onnx Elapsed time: {elapsed_time:.2f} seconds")
