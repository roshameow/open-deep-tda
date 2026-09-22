# Open Deep-TDA

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**可检查的拓扑正则化降维：C++ 持久同调核心 + PyTorch 参数化映射。**

[English](README.md) · [快速开始](#快速开始) · [相关项目](#相关项目) · [基准结果](#基准结果)

> [!IMPORTANT]
> 这是受公开 Deep-TDA 思路启发的独立实验实现，**不是** DataRefiner 官方或数值等价复现，也不宣称达到 SOTA。TopoAE 与 TopoAE++ 是各自论文作者的独立项目，只作为外部参考/基线。

<p align="center">
  <img src="assets/pipeline.svg" alt="Open Deep-TDA 处理流程" width="920">
</p>

## 功能

- C++17 Vietoris–Rips **H₀/H₁（F₂）**、确定性 MST 与关键边。
- 单形数、消元条目、消元操作和匹配规模显式预算；预算不足直接失败。
- PCA 初始化的 2D/3D PyTorch 参数化映射，支持样本外 `transform`。
- 默认距离 stress；可选 fuzzy 邻域；H₀/H₁ 与关键边目标。
- 有界分块精确近邻，或隔离进程 NN-descent + 精确 recall 审计。
- 在线拓扑子云刷新和实际样本覆盖诊断。
- 可省略训练样本的推理专用检查点。
- 离线 HTML/SVG、持久图、Mapper 摘要和命令行工具。

> [!NOTE]
> PH 只对选中的小子云精确计算，不是大规模总体上的全局拓扑保证。持久图相似也不保证语义对应或物理循环顺序正确。

真实数据实测配置可直接使用 [Fashion-MNIST PCA64 NCE](configs/fashion-pca64-nce.json) 或 [HAR 图目标](configs/har-neighborhood.json)。它们是数据集专用实验配置，不是通用默认；HAR 的 NCE 迁移没有成功改善聚类。

## 可视化示例

<p align="center">
  <img src="assets/digits-example.png" alt="Open Deep-TDA 在真实 scikit-learn Digits 数据上的嵌入，留出样本用叉号表示" width="820">
</p>

该图使用真实的 scikit-learn 内置 Digits 数据：1,400 个训练样本和 397 个通过 `transform` 映射的留出样本。圆点表示训练样本，`×` 表示留出样本；数字标签仅用于训练完成后的着色，未参与表征学习或优化。留出集 trustworthiness 为 0.864，15 邻居重叠为 0.407。此图说明样本外映射能力，不宣称优于其他方法。

### 更多真实数据示例

以下保留早期配置结果以便追溯；新的配对优化结果见下方“尚未发布版本的邻域优化”，不要把旧图理解为当前最佳可用配置。

#### UCI Human Activity Recognition

<p align="center">
  <img src="assets/har-example.png" alt="Open Deep-TDA 在 UCI HAR 官方受试者独立划分上的嵌入" width="850">
</p>

模型使用全部 **7,352 个官方训练样本**，并映射全部 **2,947 个测试样本**；训练与测试受试者不重叠。图中仅为清晰显示而进行分层抽样，活动标签只用于训练后评价。Seed 0 测试诊断：trustworthiness **0.907**、15-NN overlap **0.040**、训练后 15-NN probe 准确率 **68.7%**。

#### Fashion-MNIST

<p align="center">
  <img src="assets/fashion-mnist-example.png" alt="Open Deep-TDA 在完整 Fashion-MNIST 官方训练测试协议上的嵌入" width="850">
</p>

模型拟合全部 **60,000 张训练图像**，并映射全部 **10,000 张官方测试图像**；只有渲染图为了可读性进行分层抽样。标签在拟合后才使用。Seed 0 测试诊断：trustworthiness **0.914**、15-NN overlap **0.020**、训练后 15-NN probe 准确率 **54.1%**。

这些图来自固定运行，不是挑选最好看的随机种子。较低的邻域重叠和类别混合被如实保留；它们展示实际行为，不是营销示意图。数据署名和再分发条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装

要求 Python 3.9+、PyTorch 2.6+ 和 C++17 编译器。

```bash
git clone https://github.com/roshameow/open-deep-tda.git
cd open-deep-tda
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

可选依赖：

```bash
python -m pip install -e '.[dev]'              # 测试与 PH 对照
python -m pip install -e '.[ann]'              # NN-descent
python -m pip install -e '.[benchmark,images]' # 基准工具
```

安装过程不会自动下载数据或连接训练服务。真实数据下载必须显式启用，且不会加入仓库。

## 快速开始

```bash
deep-tda demo --dataset circle --samples 256 --features 8 \
  --config configs/quick.json --output outputs/demo
```

本地打开 `outputs/demo/report.html`。

```python
from open_deep_tda import DeepTDA
from open_deep_tda.datasets import make_dataset

X, _ = make_dataset("circle", n_samples=256, n_features=8, seed=7)
model = DeepTDA(steps=100, h1_size=32, evaluation_size=32)
Z_train = model.fit_transform(X[:200], validation_data=X[200:])
Z_new = model.transform(X[200:])

# 推理文件：不保存训练行、训练坐标和详细报告。
model.save("model.pt", include_training_data=False)
restored = DeepTDA.load("model.pt")
```

也支持数值 NPY、带 `X` 的 NPZ 和 CSV：

```bash
deep-tda fit --input train.npy --validation test.npy --output outputs/run
deep-tda transform --model outputs/run/model.pt \
  --input new.npy --output outputs/new-embedding.npy
```

CLI 输出可能包含输入数组、嵌入与完整检查点，应按数据敏感性妥善保存。紧凑检查点仍含数据训练出的权重和统计量，**不提供匿名化或差分隐私**。

## 几何目标

默认保持距离 stress：

```python
DeepTDA(geometry_objective="stress")
```

Fuzzy 邻域目标需要显式启用：

```python
DeepTDA(geometry_objective="fuzzy", fuzzy_repulsion=1.0)
```

它独立实现已有的局部亲和度与 fuzzy union 思路，不是精确 ParametricUMAP。固定 HAR 验证中，邻域重叠和标签 probe 有提升，但原始尺度 H₁ 代价明显变差，所以它仍是实验选项，未替换默认目标。

### 面向邻域保持的图目标

`geometry_objective="fuzzy_graph"` 使用**加权邻居吸引＋采样非邻居排斥**，不再在正边损失中显式排斥弱权重邻居。吸引项在每个采样批次内归一化，不是整个图比值目标的无偏估计，也不是精确 UMAP。

```python
model = DeepTDA(
    geometry_objective="fuzzy_graph", fuzzy_repulsion=1.0,
    lambda_h0=0.1, lambda_h1=0.01, steps=2400,
)
```

`residual_input_scale` 只调整神经网络残差分支的输入尺度，默认 `1.0`；**不改变参考空间距离、PCA 跳连或 PH 目标**。下面测试的按维数缩放只是特定数据上的实验，并非通用推荐。原默认配置及旧检查点行为不变。

### 条件邻居目标

`geometry_objective="neighbor_nce"` 使用 Cauchy 核分数和条件 softmax，把每个采样邻居与**同一锚点的多个非邻居**比较。排除自身、源图邻居及输入重复的负样本。训练集内部选定的是 16 个随机候选、温度 1、**不启用困难负样本挖掘**；困难挖掘没有胜出。这是独立实现的采样目标，不是精确 t-SNE、UMAP 或专有 Deep TDA。

`output_calibration="train_pairs"` 可在优化完成后，使用至多 20,000 对**仅来自 TRAIN**的样本拟合一个正输出尺度。它只改变单位，不改善邻居排名或聚类能力；尺度会保存在检查点中。结果同时保留校准前后的 PH，不能把单位校准当成拓扑形状改善的证据。

## 基准结果

审查后的聚合结果位于 [`benchmarks/results/`](benchmarks/results/)；数据、训练模型、嵌入和原始本机日志不上传。

| 协议 | Open Deep-TDA | 外部比较 | 结论 |
|---|---:|---:|---|
| UCI HAR 官方受试者划分 | 平均测试 15-NN 67.78% | UMAP 86.17%；作者 TopoAE 75.88% | 当前配置未超过成熟基线。 |
| COIL-20 已见物体的留出视角 | 准确率 72.78%；角度邻接 42.57% | UMAP 86.46%；作者 TopoAE 82.08% | 强 H₁ 条不保证物理循环顺序正确。 |
| Fashion-MNIST，训练集拟合 PCA64 | 准确率 53.90% | UMAP 77.85%；作者 TopoAE 64.89% | 不是端到端图像自监督实验。 |
| TopoAE++ 作者核心，COIL-20 | — | 准确率 81.04%；单种子 | 使用真实外部核心与明确适配，不是无改动论文复现。 |

架构与计算量并未完全匹配；测试标签只用于最终 probe。种子、协议和采样字段可查看 JSON 与 benchmark 脚本。

### 原有嵌入效果不佳的原因

- **匹配距离不等于保持邻居排名。** 即使 stress/hinge 损失为零，最近邻身份仍可能错误；反例见 [`tests/test_geometry_objective_limits.py`](tests/test_geometry_objective_limits.py)。
- **错误近邻的纠正力度不足。** 均匀非邻居采样不容易命中最易混淆的近邻对，超过全局 margin 后 hinge 不再提供排斥梯度。
- **损失目标存在竞争。** 旧 HAR/Fashion 日志中 H₀ 的嵌入梯度明显大于邻域项；这说明配比值得调整，不证明拓扑约束永远有害。
- **单纯增加训练量不够。** HAR TRAIN 内部固定受试者留出验证：stress 从 600 增至 2,400 步，邻域重叠仅 **0.0617 → 0.0659**；图目标＋弱拓扑达到 **0.0966**，残差输入调尺度后为 **0.0987**。两轮搜索均未读取类别标签或官方 TEST 数据。

全部候选（包括没有改善的变体）可通过 [`benchmarks/optimize_geometry.py`](benchmarks/optimize_geometry.py) 和 [`benchmarks/optimize_residual_scale.py`](benchmarks/optimize_residual_scale.py) 复现。先 `--preregister` 再 `--run`，两轮使用不同 `--output` 目录。这是重复使用的开发数据，并非全新的泛化证据。

### 尚未发布版本的邻域优化：固定方案确认

<p align="center">
  <img src="assets/har-optimization.png" alt="HAR 全部测试样本：固定 seed 0 的 stress 对照与仅在 TRAIN 内选出的图模型" width="1000">
</p>

**HAR，固定三个种子：**测试 15-NN 准确率 **58.23% → 82.55%**，全体候选上的 15-NN 邻域重叠 **0.0380 → 0.0775**，KMeans 测试 ARI **0.314 → 0.685**。直接在标准化 561 维参考特征上运行 KMeans，ARI 为 **0.437**。三个固定 64 点测试子集上的原始归一化 H₁ 误差 **0.002291 → 0.000901**；PCA 的原始 H₁ 误差仍更低（**0.000243**）。

本轮两组均采用相同的**仅拟合训练集的特征标准化**，与上方旧 HAR 表的预处理不同；58.23% 是新实验的配对对照，不替换历史 67.78% 结果。完整使用 7,352/2,947 个训练/测试样本。图片固定为 seed 0，段落数字为三个种子的均值；邻域指标为 256 个固定测试查询对全部训练＋测试样本计算。

配置仅依据 TRAIN 内部验证选定并锁定。HAR 测试中的无拓扑对照在邻域重叠和准确率上略好，因此这些结果**尚不能证明拓扑正则带来额外收益**；主要改善来自几何目标与优化。我们保留预先选定的弱拓扑配置，不根据 TEST 重新选择。训练预算不同（600 与 2,400 步），且官方测试数据曾在旧实验使用，并非全新外部验证集。

精确 HAR 配置见 [`configs/har-neighborhood.json`](configs/har-neighborhood.json)；所有候选、种子、原始/尺度对齐 PH 及波动统计见 [`geometry_optimization_har.json`](benchmarks/results/geometry_optimization_har.json)。复现用 [`confirm_geometry_optimization.py`](benchmarks/confirm_geometry_optimization.py)，通过 `--conditioning-pilot` 指定第二轮开发结果；注册与运行需使用相同选项。全部测试点的绘图脚本为 [`render_geometry_comparison.py`](benchmarks/render_geometry_comparison.py)。

#### 直接迁移到 Fashion-MNIST，不在 Fashion 上调参

<p align="center">
  <img src="assets/fashion-optimization.png" alt="Fashion-MNIST 全部测试样本：stress 与固定 HAR 选择配置，seed 0" width="1000">
</p>

完整 **60,000/10,000 PCA64 协议**下，三个种子的测试 15-NN 准确率 **53.90% → 64.67%**，邻域重叠 **0.0230 → 0.0431**，KMeans ARI **0.281 → 0.366**，原始归一化子集 H₁ 误差 **0.009057 → 0.002009**。直接在 PCA64 上运行 KMeans，ARI 为 **0.359**：新二维表征在这项有限聚类检查中有竞争力，但不是压倒性优势。分类准确率仍低于此前 UMAP 的 77.85%（协议/计算量不完全匹配）；原始 PCA64 参考表征的 probe 准确率仍为 85.77%。

图目标和权重直接沿用 HAR 选择结果，不搜索 Fashion 参数；残差输入采用同一维数规则 `sqrt(64)=8`。训练量从 1,200 增为 2,400 步。所有 ANN 图均通过独立精确召回率检查。原始数组与检查点保持私有。

[`configs/fashion-pca64-neighborhood.json`](configs/fashion-pca64-neighborhood.json) 用于**基准中仅拟合训练集的 PCA64 特征**，不是原始像素。完整结果见 [`geometry_optimization_fashion.json`](benchmarks/results/geometry_optimization_fashion.json)，复现脚本为 [`confirm_geometry_fashion.py`](benchmarks/confirm_geometry_fashion.py)，默认输入路径指向本地数据，不随代码发布。绘图运行 `render_geometry_comparison.py --dataset fashion_mnist`。

> [!WARNING]
> 聚类改善**不等于拓扑全面改善**。原始归一化 H₀ 误差上升：HAR **0.0594 → 0.1144**，Fashion **0.0265 → 0.0770**；较长源 H₁ 条带的未匹配比例分别从 **61.1% → 83.9%**、**27.8% → 51.7%**。图间距离更小，仍可能丢失更多长条带。这些只是固定子集诊断，不是总体拓扑保证。因此新图配置保持可选，默认 stress 不变。

### 条件邻居开发实验

新的无标签 Fashion TRAIN 内部划分为 **50,000 拟合 / 10,000 验证图像**，PCA64 仅在拟合折重新训练；不读取官方 TEST 图像或类别标签。2,400 步的验证邻域重叠为 **0.0458（上一轮图目标）→ 0.0557（邻居 NCE）**，7,200 步为 **0.0565 → 0.0677**。困难负样本挖掘与温度 0.5 均更差；全部候选保留在 [`neighbor_nce_pilot.json`](benchmarks/results/neighbor_nce_pilot.json)。

现在单独报告正边覆盖率：即使见过全部训练样本，2,400 步也只覆盖约 **66.6% 图边**；7,200 步约 **96.3%**。NCE 每步还比较更多负样本，计算更贵，不能称为等计算量的速度优势。

搜索脚本为 [`optimize_neighbor_nce.py`](benchmarks/optimize_neighbor_nce.py)，完整数据确认脚本为 [`confirm_neighbor_nce.py`](benchmarks/confirm_neighbor_nce.py)。均先 `--preregister` 再 `--run`，读取本地被忽略的数据/输出目录。模型、原始数组及私有 `docs/` 不发布。

#### NCE 固定配置完整数据确认

<p align="center">
  <img src="assets/fashion-neighbor-nce.png" alt="Fashion-MNIST 全部 TEST：上一轮图目标与条件邻居 NCE，固定 seed 0" width="1000">
</p>

**Fashion-MNIST，固定三个种子：**相对上一轮图目标，测试 15-NN 准确率 **64.67% → 68.55%**，邻域重叠 **0.0431 → 0.0565**，KMeans ARI **0.366 → 0.404**，NMI **0.529 → 0.556**。每次拟合完整 60,000 个训练样本、映射完整 10,000 个测试样本。图片固定 seed 0，段落数字为三种子均值。这是进一步改善，但仍低于此前 UMAP 的 probe 结果，不是 SOTA。

新配置使用 7,200 步，对照为 2,400 步；本轮平均拟合约 188 秒与 58 秒。两组都采用相同的 TRAIN-only 尺度校准，不能把尺度校准算作邻域/分类能力提升。未校准 H₁ 误差恶化 **0.002009 → 0.046404**；校准后约 **0.00208 → 0.00205**，但选定模型的固定子集诊断中，仍有 **92.8% 的长源 H₁ 条带未匹配**。拓扑保持问题尚未解决。

配置：[`configs/fashion-pca64-nce.json`](configs/fashion-pca64-nce.json)，输入必须是基准中仅拟合训练集的 PCA64 特征。每个种子的指标及校准前后 PH 见 [`neighbor_nce_fashion.json`](benchmarks/results/neighbor_nce_fashion.json)。不根据 TEST 重新选择；这些是重复使用的基准数据，并非全新独立泛化证据。

#### HAR 迁移没有改善聚类

<p align="center">
  <img src="assets/har-neighbor-nce.png" alt="HAR 迁移对照，固定 seed 0；三个种子的聚类效果实际退化" width="1000">
</p>

把 Fashion 选定配置锁定后迁移到 HAR，三种子准确率 **82.55% → 82.46%**，邻域重叠 **0.0775 → 0.0797**，但 KMeans ARI **0.685 → 0.614**、NMI **0.757 → 0.706**；trustworthiness 也下降。不能用 seed 0 的图片代替全部种子结论。**这不是推荐的 HAR 升级**：保留原 HAR 配置，不能假设新目标普遍更好。

校准后 H₁ 误差也从 **0.000249 → 0.000399**，两组固定诊断子集的长源 H₁ 条带均为 100% 未匹配。没有利用 HAR TEST 重新挑选其他配置。完整负面结果见 [`neighbor_nce_har.json`](benchmarks/results/neighbor_nce_har.json)；明确命名的 [`har-nce-transfer-control.json`](configs/har-nce-transfer-control.json) 仅用于复现，不作为新默认。

### v0.3 工程实测

- 正 H₁ 精简序列化仍执行相同过滤与消元。在一个 128 点 CPU 测量中，缓存 H₁ loss + backward 从 **166.8 ms 降到 117.7 ms**；这不改变 PH 最坏复杂度。
- 同一模型的推理文件由 **15,971,797 bytes 降到 75,217 bytes**，该次测试加载前后预测一致。
- 三种子固定 HAR 几何确认中，fuzzy/无拓扑得到 0.03828 邻域重叠、60.24% probe 准确率；stress/无拓扑为 0.03220、53.01%。但 raw normalized H₁ cost 约恶化 10 倍。

这些是限定协议下的测量，不是普遍性能保证。

## 相关项目

| 项目 | 与本仓库的关系 |
|---|---|
| [DataRefiner Deep TDA 文章](https://medium.com/@juanc.olamendy/deep-tda-a-new-dimensionality-reduction-algorithm-2d04fa6ed2eb) | 仅为启发来源；没有可用或复制其专有实现。 |
| [BorgwardtLab/topological-autoencoders](https://github.com/BorgwardtLab/topological-autoencoders) | Topological Autoencoders 作者实现；独立方法和外部基线。 |
| [MClemot/TopologicalAutoencodersPlusPlus](https://github.com/MClemot/TopologicalAutoencodersPlusPlus) | H₁/cascade 后续工作；不是所谓“Deep TDA 官方实现”。 |
| [danchern97/RTD_AE](https://github.com/danchern97/RTD_AE) | Representation Topology Divergence，自身采用不同 cross-complex 目标。 |
| [harishd10/TopoMap](https://github.com/harishd10/TopoMap) / [VIDA-NYU/TopoMap-pp](https://github.com/VIDA-NYU/TopoMap-pp) | 面向 H₀ 的拓扑布局，与本项目神经 H₁ 目标不同。 |
| [aidos-lab/pytorch-topological](https://github.com/aidos-lab/pytorch-topological) | 通用 PyTorch 拓扑层/损失库，不是 DataRefiner 复现。 |

使用或比较这些方法时请引用原论文/仓库。它们的许可证和数据权限不由本项目 MIT 许可证覆盖。

## 开发与测试

```bash
python -m pip install -e '.[dev]'
python -m pytest -q

cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DOPEN_DEEP_TDA_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
```

CI 覆盖 Ubuntu/macOS × Python 3.9/3.12，并测试 Python、C++、源码包/轮子内容及安装后推理。普通 CI 不要求外部研究适配器。

## 许可

原创代码采用 [MIT](LICENSE)。依赖、外部方法和数据保留各自条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。仓库不包含 DataRefiner 专有代码，也不 vendoring TopoAE/TTK 上游源码。

安全与私有数据报告见 [SECURITY.md](SECURITY.md)，贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。
