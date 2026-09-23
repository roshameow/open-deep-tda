# PH-Regularized Embedding（PH 正则化降维）

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) · [结果](#结果) · [安装](#安装) · [使用](#使用)

**PH-Regularized Embedding（PH 正则化降维）**是本项目**一条拓扑参与训练的主算法**的工作名；在当前源码中通过 `DeepTDA` 使用。仓库/包标识暂时保留 `open-deep-tda` / `open_deep_tda`。这是独立实现，**不是** DataRefiner 官方 Deep TDA，也不是它的数值等价复现。

`DeepTDA` 的训练目标结合参考空间几何、采样 Vietoris–Rips H₀/H₁ 损失和源关键边约束。有预算限制的精确 PH 由 C++ 计算；拟合后的参数化模型可 `transform` 新样本。**训练中用了 PH，不代表自动保证某条源环、全体数据或新查询的拓扑保持**，也不保证类别完全分开。

## 结果

以下对照展示的**都是实际在训练中用了 PH 的 `DeepTDA`**，不是不含 PH 的图方法替代品。标签只在无监督拟合完成后用于着色和聚类评价。图片保留**全部 TEST 点**及各格完整、独立的坐标范围，没有按标签挑点。左格是本项目主算法，其余是**外部实现**。

### Fashion-MNIST：PH 正则化降维与 UMAP

相同的仅由训练集拟合的 PCA64 输入；60,000 行拟合，10,000 行测试。固定 seed-0 图：

![Fashion-MNIST 全部测试点：左为 PH 正则化 DeepTDA，右为外部 UMAP](assets/phre-vs-external-fashion.png)

### HAR：PH 正则化降维与 UMAP

官方受试者不重叠划分：7,352 行拟合、2,947 行测试；相同训练定义的数值参考：

![HAR 全部测试点：左为 PH 正则化 DeepTDA，右为外部 UMAP](assets/phre-vs-external-har.png)

### COIL-20：PH 正则化降维、UMAP 与 TopoAE++

960 行拟合、480 行已见物体的留出视角，三种方法使用**完全相同顺序**的 PCA64 输入。TopoAE++ 是固定 seed-0 的**作者模型与损失核心＋独立 CPU 训练适配器**（1,000 次更新），不是原封不动的论文流水线或十次择优图。架构和计算预算不相同。

![COIL-20 全部测试视角：左为 PH 正则化 DeepTDA，中为外部 UMAP，右为 TopoAE++ 作者核心适配运行](assets/phre-vs-external-coil.png)

每种方法只在自身 TRAIN 布局上拟合 KMeans，再在**全部 TEST 行**评价；簇数对应数据集的类别数。同一数据集各方法输入一致，预先固定三个种子：

| 数据集 | PH 正则化降维（`DeepTDA`）TEST ARI | 外部 UMAP TEST ARI |
|---|---:|---:|
| Fashion-MNIST | .397 | **.471** |
| HAR | **.653** | .522 |
| COIL-20 | .391 | **.734** |

COIL-20 另有单独的 seed-0 三方对照：TRAIN 拟合的 **KMeans（20 簇、`n_init=10`）**得出 `DeepTDA` **.422**、UMAP **.735**、TopoAE++ 作者适配器 **.527**。这些**单种子数字不是**上表的三种子均值。更重要的是，ARI 或条形码看起来相似，**不能证明保留了同一批源环**。目前 PH 训练的主算法尚未通过 K4 上全域、同源环的验收，不能宣称拓扑保持问题已经解决。[审查后的标量结果](benchmarks/results/phre_external_comparison.json)。

## 安装

需要 Python 3.9+、C++17 编译器和 PyTorch：

```bash
git clone https://github.com/roshameow/open-deep-tda.git
cd open-deep-tda
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

安装不会自动下载基准数据。

## 使用

```python
from open_deep_tda import DeepTDA

model = DeepTDA(steps=200, h1_size=64)
Z_train = model.fit_transform(X_train)
Z_new = model.transform(X_new)
```

拟合与映射时应保持一致的数值特征定义。默认使用在训练集上定义的参考、几何 stress、非零 H₀/H₁/关键边权重；PH 在有预算限制的**子云**上计算，**不是**暗中计算任意规模的全量 Rips 复形。资源不足明确失败。上方展示的不同数据集使用了明确的专用设置，不能把构造函数默认值说成这些图片的参数；配置随[结果记录](benchmarks/results/phre_external_comparison.json)保存。训练模型可能编码或保留敏感信息，保存它不等于匿名化。

## 署名与许可

[TopoAE++](https://github.com/MClemot/TopologicalAutoencodersPlusPlus) 和 [UMAP](https://github.com/lmcinnes/umap) 是独立的外部算法，不是本项目实现。标准 PH 和拓扑自编码器思想的署名见[第三方声明](THIRD_PARTY_NOTICES.md)。原创代码采用 [MIT](LICENSE)。
