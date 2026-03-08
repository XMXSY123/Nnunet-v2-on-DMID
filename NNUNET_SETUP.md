# nnU-Net v2 环境搭建与分割

## 1. 环境准备

### 1.1 创建虚拟环境

### 1.2 安装 PyTorch
```bash

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 安装 nnU-Net v2
```bash
pip install -r requirements_nnunet.txt
```

## 2. 设置环境变量

在PyCharm终端里，先执行（路径按项目改）

**PowerShell：**
```powershell
$env:nnUNet_raw = "D:\Unet\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Unet\nnUNet_preprocessed"
$env:nnUNet_results = "D:\Unet\nnUNet_results"
```

## 3. 数据准备（80% 训练 / 20% 测试）

运行：

```bash
python prepare_nnunet_dataset.py
```
得到
- `Processed_Mammo/`：预处理图像  
- `Processed_Mammo_Masks/`：对应 PNG 遮罩  

然后运行：

```bash
python prepare_nnunet_dataset.py
```

脚本默认会：

- 只使用有遮罩的样本  
- 按 80% 训练 / 20% 测试随机划分
- 在 `nnUNet_raw` 下生成 `Dataset001_Mammo/`：imagesTr、labelsTr、imagesTs、labelsTs、dataset.json  
- **labelsTs**：20% 测试集真值标签（与 imagesTs 一一对应），用于后续算 Dice  
- 标签自动转为 0/1（背景/ROI）

可选参数示例：

```bash
python prepare_nnunet_dataset.py --train_ratio 0.8 --seed 42
python prepare_nnunet_dataset.py --output_raw D:\Unet\nnUNet_raw --dataset_name Dataset001_Mammo
```

## 4. 规划与预处理

```bash
nnUNetv2_plan_and_preprocess -d 1
```

`-d 1` 表示数据集 ID 1（Dataset**001**_Mammo）。预处理结果会写入 `nnUNet_preprocessed`。

## 5. 训练

```bash
nnUNetv2_train 1 2d 0
```

- `1`：数据集 ID  
- `2d`：2D 模型  
- `0`：折 0（nnU-Net默认 5 折，先训练折 0；如需 5 折可写脚本循环 0~4）

训练日志与权重在 `nnUNet_results` 下。

### 修改 batch size

`nnUNet_preprocessed/Dataset001_Mammo/nnUNetPlans.json` 里，`configurations` → `2d` → `batch_size`（默认如 10）。改完后直接重新训练即可，无需再跑 plan_and_preprocess。学习率不在 plans 里，只能通过自定义 Trainer 改 `initial_lr`。

## 6. 测试集推理与评估

### 6.1 推理

对 20% 测试集（imagesTs）做推理：

```bash
nnUNetv2_predict -i nnUNet_raw/Dataset001_Mammo/imagesTs -o nnUNet_pred -d 1 -c 2d -f 0
```

- `-i`：测试图像目录（imagesTs）  
- `-o`：预测结果输出目录（如 nnUNet_pred）  
- `-d 1`：数据集 1  
- `-c 2d`：2D 配置  
- `-f 0`：使用折 0 的权重  

### 6.2 结果可视化

预测图为 0/1 标签，转为「背景黑、ROI 白」的 PNG：

```bash
python vis_nnunet_pred.py
```

可选：`--pred_dir nnUNet_pred --out_dir nnUNet_pred_vis`。结果在 `nnUNet_pred_vis/`。

### 6.3 测试集 Dice

用 **labelsTs**（由 `prepare_nnunet_dataset.py` 生成）与预测对比，计算每例及平均 Dice：

```bash
python compute_test_dice.py
```

可选：`--pred_dir nnUNet_pred --labels_ts "D:\Unet\nnUNet_raw\Dataset001_Mammo\labelsTs"`，`--out_txt test_dice.txt` 可把结果写入文件。

## 7. 目录结构小结

```
Unet/
├── Processed_Mammo/          # 预处理图像
├── Processed_Mammo_Masks/    # 预处理遮罩（PNG）
├── nnUNet_raw/
│   └── Dataset001_Mammo/     # prepare_nnunet_dataset.py 生成
│       ├── imagesTr/         # 80% 训练图像
│       ├── labelsTr/         # 80% 训练标签
│       ├── imagesTs/         # 20% 测试图像
│       ├── labelsTs/         # 20% 测试真值（用于 Dice）
│       └── dataset.json
├── nnUNet_preprocessed/      # 规划与预处理输出
├── nnUNet_results/           # 训练权重与日志
├── nnUNet_pred/              # 推理输出（需自行创建或由 -o 指定）
├── nnUNet_pred_vis/          # vis_nnunet_pred.py 生成的可视化 PNG
├── prepare_nnunet_dataset.py # 80/20 划分 + 生成 Dataset001_Mammo（含 labelsTs）
├── vis_nnunet_pred.py        # 预测 0/1 转成可视图（0/255）
├── compute_test_dice.py      # 测试集 Dice（预测 vs labelsTs）
├── requirements_nnunet.txt
└── NNUNET_SETUP.md
```

## 8. 常见问题

- **找不到数据集**：检查 `nnUNet_raw` 是否设置正确，且存在 `Dataset001_Mammo` 与 `dataset.json`。  
- **PowerShell 里 set 无效**：必须用 `$env:nnUNet_raw = "..."`，不能用 CMD 的 `set`；或运行 `. .\set_nnunet_env.ps1`。  
- **预测图全黑**：预测是 0/1 标签，用 `python vis_nnunet_pred.py` 转成 0/255 再查看。
