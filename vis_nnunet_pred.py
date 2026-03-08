# -*- coding: utf-8 -*-
"""
将 nnU-Net 预测结果转为「看得见」的 PNG（背景黑、ROI 白）。
预测里像素值是 0/1，直接当 PNG 时 1 几乎全黑；这里把 1 转成 255 再存。

用法:
  python vis_nnunet_pred.py
  python vis_nnunet_pred.py --pred_dir nnUNet_pred --out_dir nnUNet_pred_vis
"""
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="把 nnU-Net 预测(0/1)转成可视化 PNG(0/255)")
    parser.add_argument("--pred_dir", type=str, default="nnUNet_pred", help="预测结果目录")
    parser.add_argument("--out_dir", type=str, default="nnUNet_pred_vis", help="可视化 PNG 输出目录")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    if not pred_dir.is_dir():
        print(f"错误: 目录不存在 {pred_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 支持 .png 和 .nii.gz
    exts = ["*.png", "*.nii.gz"]
    files = []
    for e in exts:
        files.extend(pred_dir.glob(e))

    if not files:
        print(f"在 {pred_dir} 下未找到 .png 或 .nii.gz 文件")
        return

    for f in sorted(files):
        try:
            if f.suffix == ".png" or f.suffix == "" and ".nii" in f.name:
                if f.suffix == ".png":
                    arr = np.array(Image.open(f))
                else:
                    try:
                        import nibabel as nib
                        nii = nib.load(str(f))
                        arr = np.asarray(nii.dataobj)
                        if arr.ndim > 2:
                            arr = arr.squeeze()
                    except ImportError:
                        print(f"跳过 {f.name}：.nii.gz 需安装 nibabel")
                        continue
            else:
                continue
        except Exception as e:
            print(f"读取失败 {f.name}: {e}")
            continue

        if arr.ndim == 3:
            arr = arr[:, :, 0]
        uniq = np.unique(arr)
        # 0/1 -> 0/255，便于查看
        out = (arr > 0).astype(np.uint8) * 255
        out_path = out_dir / (f.stem.replace(".nii", "") + "_vis.png")
        Image.fromarray(out).save(out_path)
        print(f"  {f.name} -> unique {uniq} -> {out_path.name}")

    print(f"已保存到 {out_dir}")


if __name__ == "__main__":
    main()
