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

## 可视化示例

<p align="center">
  <img src="assets/digits-example.png" alt="Open Deep-TDA 在真实 scikit-learn Digits 数据上的嵌入，留出样本用叉号表示" width="820">
</p>

该图使用真实的 scikit-learn 内置 Digits 数据：1,400 个训练样本和 397 个通过 `transform` 映射的留出样本。圆点表示训练样本，`×` 表示留出样本；数字标签仅用于训练完成后的着色，未参与表征学习或优化。留出集 trustworthiness 为 0.864，15 邻居重叠为 0.407。此图说明样本外映射能力，不宣称优于其他方法。

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

## 基准结果

审查后的聚合结果位于 [`benchmarks/results/`](benchmarks/results/)；数据、训练模型、嵌入和原始本机日志不上传。

| 协议 | Open Deep-TDA | 外部比较 | 结论 |
|---|---:|---:|---|
| UCI HAR 官方受试者划分 | 平均测试 15-NN 67.78% | UMAP 86.17%；作者 TopoAE 75.88% | 当前配置未超过成熟基线。 |
| COIL-20 已见物体的留出视角 | 准确率 72.78%；角度邻接 42.57% | UMAP 86.46%；作者 TopoAE 82.08% | 强 H₁ 条不保证物理循环顺序正确。 |
| Fashion-MNIST，训练集拟合 PCA64 | 准确率 53.90% | UMAP 77.85%；作者 TopoAE 64.89% | 不是端到端图像自监督实验。 |
| TopoAE++ 作者核心，COIL-20 | — | 准确率 81.04%；单种子 | 使用真实外部核心与明确适配，不是无改动论文复现。 |

架构与计算量并未完全匹配；测试标签只用于最终 probe。种子、协议和采样字段可查看 JSON 与 benchmark 脚本。

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
