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

    # 选择 provider
    providers = None
    if device and "cuda" in device.lower():
        # 如果想要在 GPU 上运行，需要安装 onnxruntime-gpu 并确保 provider 可用
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)

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
        # pad 格式：((before_dim0, after_dim0), ...)
        # 对 N,C,H,W 四维张量，只在 H 和 W 维度 pad
        pad = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))
        # np.pad 的 pad_width 需要针对每轴给出元组
        input_np = np.pad(input_np, pad, mode="reflect")

    # 运行 ONNX
    timeStart = time.time()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_np})
    timeEnd = time.time()
    elapsed_time = timeEnd - timeStart
    print(f"ONNX 推理时间: {elapsed_time:.2f} 秒")
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


    image_path = r"test_origin\train\0056_moire.jpg"
    out_path = r"results\0056_moire_out_Onnx.png"

    onnx_file = "results/0056_moire_out.onnx"

    # output_file = "C:/Codes/Moire-Zero/Moire照片/0405_moire_out_cuda.jpg"
    # record the elapsed time

    start_time = time.time()

    infer_onnx_single(onnx_file, image_path, out_path,device="cuda")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Whole Onnx Elapsed time: {elapsed_time:.2f} seconds")
