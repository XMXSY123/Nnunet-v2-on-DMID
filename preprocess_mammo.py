# -*- coding: utf-8 -*-
"""
乳腺钼靶 TIFF 图像预处理脚本
- 从黑色背景中抠出乳腺 ROI
- 翻转：使所有图像按「乳头向左」统一方向
- 填充：黑底、乳腺居中（默认画布为 max(高,宽) 的正方形）；填充后再缩放到 512×512
- 可选归一化、保存为 PNG
- 若提供 ROI_masks 目录（PNG 遮罩），会对遮罩做与图像相同的裁切、翻转、填充、缩放并保存

使用：
  1. 安装依赖：pip install -r requirements_preprocess.txt
  2. 将 TIFF 放在项目下的 TIFF Images 文件夹（可含子目录）
  3. （可选）将对应 ROI 遮罩 PNG 放在 ROI_masks 下，与 TIFF 相对路径、文件名一致（扩展名 .png）
  4. 运行：python preprocess_mammo.py
  5. 结果在 Processed_Mammo/，遮罩在 Processed_Mammo_Masks/

可选参数示例：
  --input_dir / --output_dir  输入/输出目录
  --roi_masks_dir / --output_masks_dir  ROI 遮罩（PNG）输入/输出目录
  --padding 30  裁剪 ROI 时四边预留像素
  --no_flip  不进行「乳头向左」翻转
  --resize 512 512  填充后缩放到该尺寸（默认 512×512）
  --save_with_border  保存带绿色轮廓的图便于检查（未做固定尺寸填充时有效）
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# 优先使用 tifffile 读取医学 TIFF（支持 16 位、多帧）
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
from PIL import Image


def load_mammo_tiff(path: str) -> np.ndarray:
    """加载乳腺 TIFF，统一为灰度 uint8 数组。"""
    path = str(path)
    if HAS_TIFFFILE:
        try:
            im = tifffile.imread(path)
        except Exception:
            im = np.array(Image.open(path))
    else:
        im = np.array(Image.open(path))

    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.shape[-1] == 3 else im[:, :, 0]
    if im.dtype != np.uint8:
        # 16 位或其它类型：线性拉伸到 0-255
        im = np.clip(im.astype(np.float64), 0, None)
        if im.max() > im.min():
            im = ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
    return im


def get_strip_border_box(
    gray: np.ndarray, black_max: int = 8, bright_min: int = 200, thin_dark_ratio: float = 0.88
) -> tuple:
    """计算去边后的裁切框 (y0, y1, x0, x1)，用于对图像和遮罩做相同裁切。"""
    h, w = gray.shape
    y0, y1, x0, x1 = 0, h, 0, w

    def is_black_line(vals: np.ndarray, length: int) -> bool:
        return vals.max() < black_max or vals.mean() < 3

    def is_thin_bright_line(vals: np.ndarray, length: int) -> bool:
        dark = np.sum(vals < 25) / length
        return dark >= thin_dark_ratio and vals.max() >= bright_min

    def is_border_row(y: int) -> bool:
        return is_black_line(gray[y, :], w) or is_thin_bright_line(gray[y, :], w)

    def is_border_col(x: int) -> bool:
        return is_black_line(gray[:, x], h) or is_thin_bright_line(gray[:, x], h)

    while y0 < y1 and is_border_row(y0):
        y0 += 1
    while y1 > y0 and is_border_row(y1 - 1):
        y1 -= 1
    while x0 < x1 and is_border_col(x0):
        x0 += 1
    while x1 > x0 and is_border_col(x1 - 1):
        x1 -= 1

    if y0 >= y1 or x0 >= x1:
        return 0, h, 0, w
    return y0, y1, x0, x1


def strip_border_artifacts(gray: np.ndarray, black_max: int = 8, bright_min: int = 200, thin_dark_ratio: float = 0.88) -> np.ndarray:
    """去除图像四边的非纯黑边框。"""
    y0, y1, x0, x1 = get_strip_border_box(gray, black_max, bright_min, thin_dark_ratio)
    h, w = gray.shape
    if y0 >= y1 or x0 >= x1:
        return gray
    return gray[y0:y1, x0:x1].copy()


def segment_breast_mask(gray: np.ndarray, threshold_ratio: float = 0.02) -> np.ndarray:
    """
    从灰度图中得到乳腺二值掩膜（背景为黑、乳腺为亮）。
    - 使用低阈值分离背景与乳腺，再取最大连通域并做简单形态学清理。
    """
    # 低阈值：比背景稍亮即视为前景（乳腺）
    low = max(20, int(255 * threshold_ratio))
    _, binary = cv2.threshold(gray, low, 255, cv2.THRESH_BINARY)

    # 形态学开运算去小噪点、标签等
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 取最大连通域作为乳腺
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    # 0 为背景，找面积最大的前景
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    mask = (labels == max_idx).astype(np.uint8) * 255
    return mask


def get_breast_bbox(mask: np.ndarray, padding: int = 0) -> tuple:
    """根据乳腺掩膜计算外接矩形，可加 padding（不超过图像边界）。"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    h, w = mask.shape
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + 1 + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + 1 + padding)
    return x_min, y_min, x_max, y_max


def crop_breast_roi(image: np.ndarray, mask: np.ndarray, padding: int = 0):
    """根据乳腺掩膜裁剪出乳腺 ROI 图像。返回 (roi, roi_mask, bbox)。"""
    x_min, y_min, x_max, y_max = get_breast_bbox(mask, padding)
    roi = image[y_min:y_max, x_min:x_max].copy()
    roi_mask = mask[y_min:y_max, x_min:x_max].copy()
    roi[roi_mask == 0] = 0
    return roi, roi_mask, (x_min, y_min, x_max, y_max)


def flip_roi_to_nipple_left(roi: np.ndarray, roi_mask: np.ndarray) -> tuple:
    """翻转使乳腺「乳头向左」；返回 (roi, roi_mask, did_flip)。"""
    h, w = roi.shape
    ys, xs = np.where(roi_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return roi, roi_mask, False
    cx = float(xs.mean())
    if cx < w / 2:
        return np.fliplr(roi).copy(), np.fliplr(roi_mask).copy(), True
    return roi, roi_mask, False


def pad_breast_centered(
    roi: np.ndarray,
    roi_mask: np.ndarray,
    target_height: int,
    target_width: int,
    fill_value: int = 0,
    extra: np.ndarray = None,
) -> tuple:
    """填充至固定宽高、乳腺居中。extra 为同尺寸遮罩时用相同偏移填充，返回 (padded_roi,) 或 (padded_roi, padded_extra)。"""
    rh, rw = roi.shape
    ys, xs = np.where(roi_mask > 0)
    if ys.size == 0 or xs.size == 0:
        cy, cx = rh // 2, rw // 2
    else:
        cy, cx = float(ys.mean()), float(xs.mean())
    y0 = int(round(target_height / 2 - cy))
    x0 = int(round(target_width / 2 - cx))
    src_y0, src_x0 = max(0, -y0), max(0, -x0)
    src_y1 = min(rh, target_height - y0)
    src_x1 = min(rw, target_width - x0)
    dst_y0, dst_x0 = max(0, y0), max(0, x0)
    dst_y1 = min(target_height, y0 + rh)
    dst_x1 = min(target_width, x0 + rw)

    out = np.full((target_height, target_width), fill_value, dtype=roi.dtype)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = roi[src_y0:src_y1, src_x0:src_x1]

    if extra is not None and extra.shape == roi.shape:
        out_extra = np.zeros((target_height, target_width), dtype=extra.dtype)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out_extra[dst_y0:dst_y1, dst_x0:dst_x1] = extra[src_y0:src_y1, src_x0:src_x1]
        return out, out_extra
    return (out,)


def load_roi_mask_png(path: str, ref_shape: tuple) -> np.ndarray:
    """加载 PNG 遮罩，与 ref_shape (H,W) 对齐，返回二值 uint8 数组（0/255）。"""
    im = np.array(Image.open(path))
    if im.ndim == 3:
        im = im[:, :, 0] if im.shape[-1] >= 1 else cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = (im > 0).astype(np.uint8) * 255
    if im.shape[:2] != ref_shape:
        im = cv2.resize(im, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)
        im = (im > 0).astype(np.uint8) * 255
    return im


def preprocess_one(
    input_path: str,
    output_path: str,
    *,
    threshold_ratio: float = 0.02,
    padding: int = 20,
    flip_nipple_left: bool = True,
    pad_to_square: bool = True,
    target_height: int = None,
    target_width: int = None,
    resize_to: tuple = (512, 512),
    save_with_mask_border: bool = False,
    normalize: bool = True,
    roi_mask_input_path: str = None,
    roi_mask_output_path: str = None,
) -> tuple:
    """对单张 TIFF 预处理并保存；若提供 ROI 遮罩 PNG 路径，对遮罩做相同裁切/翻转/填充/缩放并保存。返回 (成功, 是否保存了遮罩)。"""
    try:
        gray = load_mammo_tiff(input_path)
    except Exception as e:
        print(f"  [跳过] 无法读取: {input_path} -> {e}")
        return False, False

    y0, y1, x0, x1 = get_strip_border_box(gray)
    shape_before_strip = gray.shape
    roi_mask_external = None
    if roi_mask_input_path and os.path.isfile(roi_mask_input_path):
        try:
            m = load_roi_mask_png(roi_mask_input_path, shape_before_strip)
            roi_mask_external = m[y0:y1, x0:x1].copy()
        except Exception as e:
            print(f"  [警告] 无法加载 ROI 遮罩: {roi_mask_input_path} -> {e}，跳过遮罩输出")
            roi_mask_external = None
    gray = gray[y0:y1, x0:x1].copy()

    mask = segment_breast_mask(gray, threshold_ratio=threshold_ratio)
    roi, roi_mask, (x_min, y_min, x_max, y_max) = crop_breast_roi(gray, mask, padding=padding)

    if roi.size == 0:
        print(f"  [跳过] 未检测到乳腺区域: {input_path}")
        return False, False

    if roi_mask_external is not None:
        roi_mask_external = roi_mask_external[y_min:y_max, x_min:x_max].copy()

    if flip_nipple_left:
        roi, roi_mask, did_flip = flip_roi_to_nipple_left(roi, roi_mask)
        if roi_mask_external is not None and did_flip:
            roi_mask_external = np.fliplr(roi_mask_external).copy()

    do_pad = pad_to_square or (target_height is not None and target_width is not None and target_height > 0 and target_width > 0)
    if do_pad:
        if target_height is not None and target_width is not None and target_height > 0 and target_width > 0:
            th, tw = target_height, target_width
        else:
            th, tw = max(roi.shape[0], roi.shape[1]), max(roi.shape[0], roi.shape[1])
        if roi_mask_external is not None:
            roi, roi_mask_external = pad_breast_centered(roi, roi_mask, th, tw, fill_value=0, extra=roi_mask_external)
        else:
            roi = pad_breast_centered(roi, roi_mask, th, tw, fill_value=0)[0]

    if resize_to and len(resize_to) == 2 and resize_to[0] > 0 and resize_to[1] > 0:
        roi = cv2.resize(roi, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_AREA)
        if roi_mask_external is not None:
            roi_mask_external = cv2.resize(roi_mask_external, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_NEAREST)
            roi_mask_external = (roi_mask_external > 0).astype(np.uint8) * 255

    if normalize:
        roi = np.clip(roi.astype(np.float64), 0, None)
        if roi.max() > roi.min():
            roi = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(np.uint8)
        else:
            roi = roi.astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if save_with_mask_border and not do_pad:
        contour_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(output_path, contour_img)
    else:
        cv2.imwrite(output_path, roi)

    mask_saved = False
    if roi_mask_output_path and roi_mask_external is not None:
        d = os.path.dirname(roi_mask_output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        cv2.imwrite(roi_mask_output_path, roi_mask_external)
        mask_saved = True
    return True, mask_saved


def main():
    parser = argparse.ArgumentParser(description="乳腺钼靶 TIFF 预处理：抠出乳腺 ROI 并保存")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "TIFF Images"),
        help="存放 TIFF 的根目录（会递归搜索 .tif .tiff）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Processed_Mammo"),
        help="预处理结果保存目录",
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.02,
        help="前景阈值（相对 255 的比例），用于分离背景与乳腺，默认 0.02",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="裁剪 ROI 时四边预留像素",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="不进行「乳头向左」翻转（默认会翻转使方向一致）",
    )
    parser.add_argument(
        "--no_pad",
        action="store_true",
        help="不进行填充（默认会将宽高补齐为正方形，如 1000*3000 -> 3000*3000，黑底乳腺居中）",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=None,
        help="填充画布高度（与 --target_width 同时指定时使用；否则自动为 max(高,宽)）",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="填充画布宽度（与 --target_height 同时指定时使用）",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="在填充好的图像上缩放到该尺寸，默认 512 512（即 512×512）",
    )
    parser.add_argument(
        "--save_with_border",
        action="store_true",
        help="保存带乳腺轮廓的图（便于检查），否则只保存裁剪后的灰度图",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="不做 min-max 归一化",
    )
    parser.add_argument(
        "--roi_masks_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ROI_masks"),
        help="ROI 遮罩目录（PNG），与 TIFF 相对路径一致",
    )
    parser.add_argument(
        "--output_masks_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Processed_Mammo_Masks"),
        help="预处理遮罩保存目录",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    roi_masks_dir = Path(args.roi_masks_dir) if args.roi_masks_dir else None
    output_masks_dir = Path(args.output_masks_dir) if args.output_masks_dir else None

    suffixes = (".tif", ".tiff")
    seen = set()
    files = []
    for suf in suffixes:
        for f in input_dir.rglob(f"*{suf}"):
            if f.resolve() not in seen:
                seen.add(f.resolve())
                files.append(f)
    files = sorted(files, key=lambda p: str(p))

    if not files:
        print(f"在 {input_dir} 下未找到任何 TIFF 文件。")
        return

    def find_roi_mask_png(tiff_path: Path, masks_root: Path) -> str:
        """按 TIFF 相对路径在 masks 目录下查找同名 .png 遮罩。"""
        try:
            rel = tiff_path.relative_to(input_dir)
        except ValueError:
            rel = Path(tiff_path.name)
        p = masks_root / rel.parent / (rel.stem + ".png")
        return str(p) if p.is_file() else ""

    print(f"共找到 {len(files)} 张 TIFF，开始预处理（抠出乳腺 ROI）...")
    if roi_masks_dir and roi_masks_dir.is_dir() and output_masks_dir:
        output_masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"ROI 遮罩（PNG）: {roi_masks_dir} -> {output_masks_dir}")
    success = 0
    masks_saved = 0
    for i, f in enumerate(files):
        rel = f.relative_to(input_dir) if input_dir in f.parents else f.name
        out_path = output_dir / (rel.with_suffix(".png"))
        roi_in = find_roi_mask_png(f, roi_masks_dir) if roi_masks_dir else ""
        roi_out = str(output_masks_dir / (rel.with_suffix(".png"))) if output_masks_dir and roi_in else ""
        ok, mask_saved = preprocess_one(
            str(f),
            str(out_path),
            threshold_ratio=args.threshold_ratio,
            padding=args.padding,
            flip_nipple_left=not args.no_flip,
            pad_to_square=not args.no_pad,
            target_height=args.target_height,
            target_width=args.target_width,
            resize_to=tuple(args.resize) if args.resize else None,
            save_with_mask_border=args.save_with_border,
            normalize=not args.no_normalize,
            roi_mask_input_path=roi_in or None,
            roi_mask_output_path=roi_out or None,
        )
        if ok:
            success += 1
        if mask_saved:
            masks_saved += 1
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(files)}")
    print(f"完成：成功 {success}/{len(files)}，图像保存在 {output_dir}")
    if output_masks_dir is not None:
        print(f"  遮罩已保存 {masks_saved} 张到 {output_masks_dir}" if masks_saved else f"  未保存遮罩（请确认 {roi_masks_dir} 下存在与 TIFF 同路径的 .png 文件）")


if __name__ == "__main__":
    main()
