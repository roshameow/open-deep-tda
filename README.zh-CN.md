# Open Deep-TDA

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) · [安装](#安装) · [示例](#示例) · [结果](#结果)

受公开 Deep-TDA 思路启发的独立、可检查降维项目。**不是 DataRefiner 官方实现，也不是数值等价复现。** 当前图降维器不保证类别分离或源环保持。旧版 `DeepTDA` 神经估计器及有预算限制的结构验证工具仍可使用。

## 结果

**哪张是本项目的结果？** 下方 K4 对照图的**中间一张“GraphEmbedding”就是 Open Deep-TDA**；左侧是外部 TopoAE++ 作者核心的 CPU 适配运行，右侧是 UMAP。图中显示全部 300 个输入点。颜色表示源数据行序，**不是类别**。三者计算预算不同；作者适配器仅运行一个固定种子，不是论文的十次择优图。条带数量相似也不足以证明保持了*同一组*源环。

![K4：左侧 TopoAE++ 适配器，中间本项目 GraphEmbedding，右侧 UMAP](assets/visual-contracts-K4.png)

当前图布局在 K4 上的显著 H₁ 条带少于本次 TopoAE++ 适配运行（三条对一条）。这是当前结果的局限，不能称为拓扑保持成功。

### Digits：两种输入表示、两种降维器

下图**左侧两张都是本项目的结果**：左上是原始像素输入的 Open Deep-TDA 预计算图适配器；左下是显式选用平移切空间图像差异度后，由**同一图适配器**得到的布局。右侧两张分别是相同输入上的外部 UMAP。每一格显示全部 1,797 个 Digits 点；数字类别仅在拟合后用于着色。这是传导式展示，不是留出集验证。

![Digits：左列是 Open Deep-TDA 预计算图，右列是 UMAP；上排原始像素，下排实验性平移切空间差异度](assets/digits-image-geometry.png)

| 输入；全部 1,797 个 Digits；seed 0 | 本项目预计算图：KMeans10 ARI | UMAP：KMeans10 ARI |
|---|---:|---:|
| 原始像素 | .822 | .822 |
| 平移切空间差异度 | .848 | .904 |

此图像差异度使**两种方法**在本例中获益，并不证明我们的优化器优于 UMAP。它不是严格度量或精确平移不变性；它改变了输入拓扑，**不是**一般数值向量的默认参考。固定训练／留出 Digits 划分上的结果有升有降，部分数字仍被拆成多个簇。

另一个固定的 Fashion-MNIST 图方法对照（三种子，60,000 TRAIN / 10,000 TEST）中，本项目拟合后 KMeans ARI **.421**，低于 UMAP 的 **.471**；拟合后 15-NN 准确率则是 **78.49%** 对 **77.88%**。分类准确率不能替代聚类布局验收。[基准聚合结果](benchmarks/results/graph_core_compact_confirmation.json)。

## 安装

要求 Python 3.9+ 与 C++17 编译器。神经基线另需 PyTorch；图与图像示例依赖以下可选包。

```bash
git clone https://github.com/roshameow/open-deep-tda.git
cd open-deep-tda
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[graph,plot,images]'
```

此处的图／图像功能位于**尚未发布的 main**，不在旧 v0.3.0 发布包中。安装不会自动下载基准数据。

## 示例

```python
from open_deep_tda import GraphEmbedding

model = GraphEmbedding(seed=0)
Z_train = model.fit_transform(X_train)  # 有限数值数组，至少16行
Z_new = model.transform(X_new)         # 与训练输入相同的特征定义
```

`GraphEmbedding` 包含源近邻图、坐标优化及条件式样本外映射；**不保证 H₀/H₁ 保持**。预测器保留训练特征，保存的模型不能视为匿名化数据。

可选的小型图像路线会显式改变输入差异度：

```python
from open_deep_tda import TranslationTangentDissimilarity, PrecomputedGraphEmbedding

reference = TranslationTangentDissimilarity()
D_train = reference.fit_transform(images)  # 灰度图 (N,H,W)
model = PrecomputedGraphEmbedding()
Z_train = model.fit_transform(D_train)
Z_new = model.transform(reference.transform(new_images))
```

稠密图适配器默认最多接受 2,000 个参考图像；查询距离矩阵的列必须与训练行顺序严格对应。此实验适配器没有检查点持久化接口，也不保证样本外效果。拟合后的参考对象保留训练图像副本。

```bash
python examples/run_graph_embedding.py --plot
python examples/digits_image_geometry.py --umap  # --umap 需安装 umap-learn
```

输出位于本地被忽略的 `outputs/`；公开坐标或模型前应检查数据许可。历史神经基线可通过 `from open_deep_tda import DeepTDA` 使用。结构检查器和构造布局是独立的可选 API；图结果不会自动继承结构证书。

## 署名与许可

[TopoAE++](https://github.com/MClemot/TopologicalAutoencodersPlusPlus)、[TopoAE](https://github.com/BorgwardtLab/topological-autoencoders)、[UMAP](https://github.com/lmcinnes/umap)、[TopoMap](https://github.com/harishd10/TopoMap) 均为独立项目。TopoAE++ 图来自外部作者核心适配运行，其代码未打包进本仓库。依赖与数据署名见[第三方声明](THIRD_PARTY_NOTICES.md)。本项目原创代码采用 [MIT](LICENSE)。
