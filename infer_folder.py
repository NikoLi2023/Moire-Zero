import argparse
import time
from pathlib import Path
from typing import Iterable, List, Set

import cv2

from infer_single import infer_single


DEFAULT_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def parse_exts(exts_text: str) -> Set[str]:
    exts = set()
    for part in exts_text.split(","):
        e = part.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        exts.add(e)
    return exts if exts else set(DEFAULT_EXTS)


def collect_images(input_dir: Path, recursive: bool, exts: Set[str]) -> List[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def build_output_path(
    in_file: Path,
    input_dir: Path,
    output_dir: Path,
    out_ext: str = "",
) -> Path:
    rel = in_file.relative_to(input_dir)
    out_file = output_dir / rel
    if out_ext:
        ext = out_ext if out_ext.startswith(".") else "." + out_ext
        out_file = out_file.with_suffix(ext.lower())
    return out_file


def main():
    input_dir = Path(r"D:\Raw Data 8Bit").resolve()
    output_dir = Path(r"D:\Remove Moire Demo").resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    images = collect_images(input_dir,True,DEFAULT_EXTS)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found images: {len(images)}")

    if len(images) == 0:
        return
    start_all = time.time()
    ok = 0
    fail = 0
    for idx, in_file in enumerate(images, 1):
        out_file = build_output_path(in_file, input_dir, output_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(images)}] Infer: {in_file}")

        try:
            infer_single(
                ckpt_path=r"results/test_origin/ckpt/200.pth.tar",
                image_path=str(in_file),
                out_path=str(out_file),
                config_filename="UHDM_m_1.yml",
                device="cuda:0",
                save_onnx=False,
                onnx_path=None,
                using_fp16=False,
            )
            ok += 1

            #covert out_path to one channel file
            src=cv2.imread(str(out_file), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out_file), src)
        except Exception as e:
            fail += 1
            print(f"  -> Failed: {in_file}")
            print(f"  -> Error: {e}")


    elapsed = time.time() - start_all
    print("=" * 60)
    print(f"Done. success={ok}, fail={fail}, total={len(images)}, elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
