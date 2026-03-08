# -*- coding: utf-8 -*-
"""
将 Processed_Mammo + Processed_Mammo_Masks 中有遮罩的样本按 80% 训练 / 20% 测试
整理为 nnU-Net v2 所需格式（Dataset001_Mammo），并生成 dataset.json。

使用前请设置环境变量（或在下方修改默认路径）:
  nnUNet_raw   - 原始数据集根目录，脚本会在其下创建 Dataset001_Mammo/
  nnUNet_preprocessed
  nnUNet_results

使用:
  1. 已运行 preprocess_mammo.py，得到 Processed_Mammo/ 与 Processed_Mammo_Masks/
  2. 设置环境变量后执行: python prepare_nnunet_dataset.py
  3. 可选: --images_dir / --masks_dir / --output_raw / --seed / --train_ratio
"""

import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="整理乳腺 ROI 图像与遮罩为 nnU-Net v2 数据集（80%% 训练 / 20%% 测试）")
    parser.add_argument(
        "--images_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Processed_Mammo"),
        help="预处理图像目录（与遮罩同相对路径）",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Processed_Mammo_Masks"),
        help="预处理遮罩目录",
    )
    parser.add_argument(
        "--output_raw",
        type=str,
        default=None,
        help="nnUNet 原始数据根目录，默认用环境变量 nnUNet_raw 或 项目/nnUNet_raw",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset001_Mammo",
        help="数据集文件夹名，如 Dataset001_Mammo",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="用于训练的比例（默认 0.8，即 80%% 训练、20%% 测试）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="种子",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    if not images_dir.is_dir():
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    if not masks_dir.is_dir():
        print(f"错误: 遮罩目录不存在: {masks_dir}")
        return

    # 收集「有遮罩」的样本：按相对路径配对
    pairs = []
    for im_path in images_dir.rglob("*.png"):
        rel = im_path.relative_to(images_dir)
        mask_path = masks_dir / rel
        if mask_path.is_file():
            pairs.append((str(im_path), str(mask_path), rel))

    if not pairs:
        print(f"在 {images_dir} 与 {masks_dir} 下未找到任何成对图像与遮罩。")
        return

    # 80% 训练 / 20% 测试
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(pairs))
    n_train = max(1, int(len(pairs) * args.train_ratio))
    train_idx = set(indices[:n_train])
    test_idx = set(indices[n_train:])

    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    print(f"共 {len(pairs)} 个有遮罩样本，训练 {len(train_pairs)}，测试 {len(test_pairs)}")

    # 输出目录：nnUNet_raw/Dataset001_Mammo/
    raw_root = args.output_raw or os.environ.get("nnUNet_raw")
    if not raw_root:
        raw_root = os.path.join(os.path.dirname(__file__), "nnUNet_raw")
    raw_root = Path(raw_root)
    dataset_dir = raw_root / args.dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    # nnU-Net v2 命名：case_0000_0000.png（图像，通道 0000）, case_0000.png（标签）
    file_ending = ".png"

    def save_label_01(mask_path: str, out_path: Path):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)
        Image.fromarray(mask).save(out_path)

    def copy_and_name(case_id: str, im_path: str, mask_path: str, is_train: bool):
        im_name = f"{case_id}_0000{file_ending}"
        label_name = f"{case_id}{file_ending}"
        if is_train:
            shutil.copy2(im_path, images_tr / im_name)
            save_label_01(mask_path, labels_tr / label_name)
        else:
            shutil.copy2(im_path, images_ts / im_name)
            save_label_01(mask_path, labels_ts / label_name)

    for i, (im_path, mask_path, rel) in enumerate(train_pairs):
        case_id = f"mammo_{i:04d}"
        copy_and_name(case_id, im_path, mask_path, is_train=True)

    for i, (im_path, mask_path, rel) in enumerate(test_pairs):
        case_id = f"mammo_test_{i:04d}"
        copy_and_name(case_id, im_path, mask_path, is_train=False)

    # dataset.json（2D 单通道灰度）
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "roi": 1},
        "numTraining": len(train_pairs),
        "file_ending": file_ending,
    }
    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"已写入: {dataset_dir}")
    print(f"  imagesTr: {len(train_pairs)} 例, labelsTr: {len(train_pairs)} 例")
    print(f"  imagesTs: {len(test_pairs)} 例, labelsTs: {len(test_pairs)} 例（20% 测试集真值标签）")
    print(f"请设置环境变量 nnUNet_raw={raw_root.resolve()}，然后运行 nnUNetv2_plan_and_preprocess 与 nnUNetv2_train。")
    if not os.environ.get("nnUNet_raw"):
        print(f"建议: set nnUNet_raw={raw_root.resolve()}  或  export nnUNet_raw={raw_root.resolve()}")


if __name__ == "__main__":
    main()
