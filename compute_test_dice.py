# -*- coding: utf-8 -*-
"""
计算测试集 Dice：对比 nnU-Net 预测与 labelsTs 真值，输出每例及平均 Dice。

用法:
  python compute_test_dice.py
  python compute_test_dice.py --pred_dir nnUNet_pred --labels_ts nnUNet_raw/Dataset001_Mammo/labelsTs
"""
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    """加载掩膜为二值 0/1。支持 .png 和 .nii.gz。"""
    path = Path(path)
    if not path.is_file():
        return None
    if path.suffix.lower() == ".png":
        arr = np.array(Image.open(path))
    else:
        try:
            import nibabel as nib
            arr = np.asarray(nib.load(str(path)).dataobj)
        except Exception:
            return None
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim > 2:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """二值 Dice（前景类）。pred、gt 为 0/1。"""
    pred = pred.ravel().astype(bool)
    gt = gt.ravel().astype(bool)
    if not gt.any():
        return 1.0 if not pred.any() else 0.0
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="计算测试集 Dice（预测 vs labelsTs）")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="nnUNet_pred",
        help="预测结果目录（与 labelsTs 文件名对应，如 mammo_test_0000.png / .nii.gz）",
    )
    parser.add_argument(
        "--labels_ts",
        type=str,
        default=None,
        help="测试集真值目录，默认 nnUNet_raw/Dataset001_Mammo/labelsTs",
    )
    parser.add_argument(
        "--out_txt",
        type=str,
        default="",
        help="可选：将每例 Dice 与均值写入该文本文件",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if args.labels_ts:
        labels_ts = Path(args.labels_ts)
    else:
        raw = os.environ.get("nnUNet_raw") or os.path.join(os.path.dirname(__file__), "nnUNet_raw")
        labels_ts = Path(raw) / "Dataset001_Mammo" / "labelsTs"

    if not pred_dir.is_dir():
        print(f"错误: 预测目录不存在 {pred_dir}")
        return
    if not labels_ts.is_dir():
        print(f"错误: 真值目录不存在 {labels_ts}")
        return

    # 收集 labelsTs 里的 case id（mammo_test_0000.png -> mammo_test_0000）
    gt_files = {}
    for f in labels_ts.glob("*.png"):
        stem = f.stem
        if stem.endswith("_0000"):
            stem = stem[:-5]
        gt_files[stem] = f
    for f in labels_ts.glob("*.nii.gz"):
        stem = f.name.replace(".nii.gz", "")
        if stem.endswith("_0000"):
            stem = stem[:-5]
        gt_files[stem] = f

    if not gt_files:
        print(f"在 {labels_ts} 下未找到 .png 或 .nii.gz 标签文件")
        return

    # 对每个 case 找预测文件（nnU-Net 可能存成 case.png 或 case.nii.gz）
    results = []
    for case_id, gt_path in sorted(gt_files.items()):
        pred_path = pred_dir / f"{case_id}.png"
        if not pred_path.is_file():
            pred_path = pred_dir / f"{case_id}.nii.gz"
        if not pred_path.is_file():
            pred_path = pred_dir / f"{case_id}_0000.png"
        if not pred_path.is_file():
            print(f"  跳过 {case_id}: 未找到预测文件")
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        if gt is None or pred is None:
            print(f"  跳过 {case_id}: 读取失败")
            continue
        if gt.shape != pred.shape:
            from PIL import Image as PILImage
            pred = np.array(
                PILImage.fromarray((pred > 0).astype(np.uint8) * 255).resize(
                    (gt.shape[1], gt.shape[0]), resample=PILImage.NEAREST
                )
            )
            pred = (pred > 0).astype(np.uint8)

        d = dice(pred, gt)
        results.append((case_id, d))
        print(f"  {case_id}: Dice = {d:.4f}")

    if not results:
        print("没有成功匹配的预测/真值对")
        return

    names, dices = zip(*results)
    mean_dice = float(np.mean(dices))
    print(f"\n测试集平均 Dice: {mean_dice:.4f}  (共 {len(results)} 例)")

    if args.out_txt:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for name, d in results:
                f.write(f"{name}\t{d:.4f}\n")
            f.write(f"mean\t{mean_dice:.4f}\n")
        print(f"已写入: {args.out_txt}")


if __name__ == "__main__":
    main()
