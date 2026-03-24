# 使用 CIFAR-10 数据集的 MoE Tutorial 实验记录

一个面向入门与实践的 MoE（Mixture of Experts）小项目。这个仓库记录了围绕 **CIFAR-10 图像分类** 做的几组实验：从普通 CNN baseline 出发，逐步过渡到带有路由机制的 MoE 结构，并继续把专家形式从“hidden feature mixing”推进到更接近原生 FFN-MoE 的实现。


## 1. 工作主线

工作主线概括为三步：

### Step 1：建立 baseline
先在 CIFAR-10 上实现并训练一个普通 CNN 分类器，作为后续对比的起点。

### Step 2：加入 MoE 机制（HiddenMix 版本）
在卷积特征提取之后，引入门控网络（gate）和多个专家（experts），先让专家输出 hidden representation，再对 Top-k 专家的结果进行加权融合，最后送入分类层。

围绕这个工作展开以下尝试：
- 让 **MoE 路由机制** 成功运行；
- 观察不同类别对不同专家的偏好；
- 解决训练初期容易出现的 **专家坍缩 / 路由偏科** 问题。

### Step 3：改成更贴近原生 MoE 的 FFNMix 版本
在前一个版本取得的稳健参数基础上，进一步把专家改写成更接近 Transformer 中 FFN 的形式：
- 每个专家不再只输出 hidden；
- 而是直接完成一套更完整的前馈映射；
- 最终由 gate 对多个专家输出进行 Top-k 加权融合。


---

## 2. 实验结果概览

目前仓库中已经保留了三组关键结果：

| Model | Best Test Accuracy | Best Epoch | 说明 |
|---|---:|---:|---|
| CIFAR-10 Baseline CNN | **86.75%** | 58 | 普通 CNN 基线 |
| CIFAR-10 MoE (Soft HiddenMix) | **88.11%** | 60 | 专家输出 hidden，再融合 |
| CIFAR-10 MoE (FFNMix) | **89.79%** | 57 | 更贴近原生 FFN-MoE |


---

## 3. 和baseline相比在MoE相关内容上的工作以及部分超参


### 3.1 Warmup 机制
相比于最原生的 MoE，在训练最开始的若干 epoch 中，这个模型不立即完全依赖路由器做强选择，而是让训练过程先更平稳地启动，避免 gate 在最早期就把流量过度压到少数专家上。
这个损招是AI想的，我本来还不知道，专家坍缩的问题一直没法解决，逼急了试了下，出奇地好用。

当前实验中：
- `warmup_epochs = 10`

### 3.2 Top-k 路由
不是所有样本都发送到全部专家，而是只选择分数最高的 `k` 个专家参与计算。

当前实验中：
- `num_experts = 4`
- `top_k = 2`

这既保留了条件计算的思想，也让结构更像真实 MoE。

### 3.3 Gate Temperature
通过 temperature 调整门控分布的“软硬程度”，让专家选择不至于过早过尖锐。

当前实验中：
- `gate_temperature = 1.2`

### 3.4 Gate Noise
在训练早期对 gate logits 注入更强噪声，后期再减小噪声，帮助探索并减少早期专家垄断。

当前实验中：
- 初期噪声：`0.12`
- 后期噪声：`0.03`
- 切换 epoch：`28`

### 3.5 Balance Loss
额外加入负载均衡项，鼓励不同专家都能接收到一定比例的样本，而不是只让极少数专家长期工作。

当前实验中：
- `balance_loss_weight = 0.25`


---

## 4. 类别 - 专家偏好现象

在 `07_cifar10_moe_soft_hiddenmix.py` 的实验结果中，额外保存了一个 `class -> expert preference` 分析文件，观察每个类别更偏向哪些专家。

可以证明，专家训练后已经出现了一定程度的**语义分工倾向**。虽然这还不是特别强、特别“纯”的专家专精，但已经能说明 gate 确实在学习“不同类别适合交给不同专家”这件事。

---

## 5. 仓库结构

```text
moe_tutorial/
├── 01_check_gpu.py
├── 04_fashionmnist_baseline.py
├── 05_fashionmnist_moe.py
├── 06_cifar10_baseline.py
├── 07_cifar10_moe_soft_hiddenmix.py
├── 08_cifar10_moe_ffnmix.py
├── data/
└── output/
    ├── CIFAR-10 baseline accuracy.png
    ├── CIFAR-10 baseline loss.png
    ├── CIFAR-10 MoE accuracy.png
    ├── CIFAR-10 MoE Loss.png
    ├── cifar10_baseline_training_metrics.txt
    ├── cifar10_moe_training_metrics.txt
    ├── cifar10_moe_ffnmix_training_metrics.txt.txt
    ├── cifar10_moe_class_expert_preference.txt
    ├── cifar10_moe_ffnmix_accuracy_curve.png
    ├── cifar10_moe_ffnmix_loss_curve.png
    ├── cifar10_moe_samples.png
    └── cifar10_moe_ffnmix_best_model.pth
```

---

## 6. 各脚本说明

### `01_check_gpu.py`
用于检查当前 PyTorch / CUDA / GPU 环境是否正常。


### `06_cifar10_baseline.py`
CIFAR-10 的普通 CNN baseline，也是后续 MoE 对比的基础。


### `07_cifar10_moe_soft_hiddenmix.py`
CIFAR-10 上的 HiddenMix MoE 版本：
- 卷积 backbone 提特征；
- 多个专家输出 hidden representation；
- gate 选择 Top-k 专家并加权融合；
- 再送入分类头。

### `08_cifar10_moe_ffnmix.py`
CIFAR-10 上的 FFNMix MoE 版本：
- 专家更像完整 FFN；
- 比 HiddenMix 更贴近原生 MoE / Transformer-FFN 替换思路；
- 也是当前仓库里效果最好的版本。

---

## 7. 如何运行

### 环境依赖
建议使用 Python 3.10+，并安装以下核心依赖：

```bash
pip install torch torchvision matplotlib
```

### 数据集
当前脚本默认从本地读取数据，不依赖联网下载。路径里主要使用：

```python
DATA_ROOT = "/mnt/data"
```

如果不是在阿里云开发机或类似环境中运行，可以把脚本中的 `DATA_ROOT` 改成自己的本地数据路径。

### 运行示例

检查 GPU：

```bash
python 01_check_gpu.py
```

运行 CIFAR-10 baseline：

```bash
python 06_cifar10_baseline.py
```

运行 HiddenMix MoE：

```bash
python 07_cifar10_moe_soft_hiddenmix.py
```

运行 FFNMix MoE：

```bash
python 08_cifar10_moe_ffnmix.py
```

---

## 8. 输出结果说明

训练过程中，脚本会在 `output/` 目录下保存：

- accuracy 曲线
- loss 曲线
- 训练指标文本
- 类别到专家偏好的分析结果


---

