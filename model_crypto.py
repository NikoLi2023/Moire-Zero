# -*- coding: utf-8 -*-
"""训练后 ONNX 模型加密。

通过 pythonnet (import clr) 调用同目录下 ModelCrypto.dll（C#）的静态方法：
    ModelCrypto.ModelCrypto.EncryptFunc(string inputModelFile, string outputModelFile)

约定：
  1. 加密产物命名为 "<stem>.demoire.qgAi"；
  2. 加密成功后删除原 .onnx；
  3. 加密失败（DLL 缺失 / pythonnet 异常 / EncryptFunc 返回 False）同样删除原 .onnx，
     并把失败原因打印到日志 —— 调用方（quick_train.py，经 train_ui.py 的 DailyLogWriter）
     负责把输出落到 Log/ 按天日志文件；
  4. ModelCrypto.dll 查找顺序：onnx 所在目录 -> 本脚本所在目录（项目根）。

依赖：pip install pythonnet
"""
import os

_DLL_NAME = 'ModelCrypto'
_crypto_type = None


def _find_dll(onnx_path):
    candidates = [
        os.path.dirname(os.path.abspath(onnx_path)),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for d in candidates:
        p = os.path.join(d, _DLL_NAME + '.dll')
        if os.path.isfile(p):
            return p
    return None


def _load_crypto(dll_path):
    """加载并返回 C# ModelCrypto 类型（带缓存）。失败抛异常。"""
    global _crypto_type
    if _crypto_type is not None:
        return _crypto_type
    import clr  # pythonnet
    # 按文件名加载最稳妥：把 DLL 所在目录加入 sys.path 后按程序集名引用
    import sys
    dll_dir = os.path.dirname(dll_path)
    if dll_dir not in sys.path:
        sys.path.append(dll_dir)
    clr.AddReference(_DLL_NAME)
    from ModelCrypto import ModelCrypto  # namespace ModelCrypto, class ModelCrypto
    _crypto_type = ModelCrypto
    return _crypto_type


def encrypt_onnx_and_remove(onnx_path):
    """加密 onnx -> <stem>.gqAi，然后删除原 onnx（无论加密成败）。

    返回 True/False；过程信息全部 print，由调用方落盘到 Log。
    """
    onnx_path = os.path.abspath(onnx_path)
    stem, _ = os.path.splitext(onnx_path)
    enc_path = stem + '.demoire.qgAi'

    if not os.path.isfile(onnx_path):
        print(f'[Crypto] 待加密文件不存在，跳过：{onnx_path}')
        return False

    ok, err = False, None
    dll_path = _find_dll(onnx_path)
    if dll_path is None:
        err = f'未找到 {_DLL_NAME}.dll（已查找 onnx 所在目录与脚本目录）'
    else:
        try:
            crypto = _load_crypto(dll_path)
            ok = bool(crypto.EncryptFunc(onnx_path, enc_path))
            if not ok:
                err = 'EncryptFunc 返回 False'
        except Exception as e:  # pythonnet 加载失败 / .NET 异常等
            err = f'{type(e).__name__}: {e}'

    # 按约定：无论加密成功与否，都删除原 onnx
    try:
        os.remove(onnx_path)
    except OSError as e:
        print(f'[Crypto] 删除原 onnx 失败：{onnx_path}（{e}）')

    if ok and os.path.isfile(enc_path):
        print(f'Model Encrypted: {enc_path}')
        return True

    print(f'[Crypto] 加密失败：{err}；原 onnx 已按策略删除：{onnx_path}')
    return False


if __name__ == '__main__':
    # 手动验证：python model_crypto.py <某个.onnx>
    import sys
    if len(sys.argv) < 2:
        print('用法: python model_crypto.py <onnx文件路径>')
        sys.exit(1)
    encrypt_onnx_and_remove(sys.argv[1])
