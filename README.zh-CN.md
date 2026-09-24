# PH-Regularized Embedding（PH 正则化降维）

[![Tests](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml/badge.svg)](https://github.com/roshameow/open-deep-tda/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) · [结果](#结果) · [安装](#安装) · [使用](#使用)

**PH-Regularized Embedding（PH 正则化降维）**是本项目**一条拓扑参与训练的主算法**的工作名；在当前源码中通过 `DeepTDA` 使用。仓库/包标识暂时保留 `open-deep-tda` / `open_deep_tda`。这是独立实现，**不是** DataRefiner 官方 Deep TDA，也不是它的数值等价复现。

`DeepTDA` 的训练目标结合参考空间几何、采样 Vietoris–Rips H₀/H₁ 损失和源关键边约束；有预算限制的精确 PH 由 C++ 计算，拟合后的模型可 `transform`。对于**事先仅从源数据选定的环契约**，显式启用 `fit_with_topology_guidance` 会继续使用**同一个已经使用 PH 的模型**及经验证的源引导平面布局。如果 teacher 尚未通过最终验收，默认训练路径会按需加入完整 TRAIN 的 H₀ 连接约束和精确 F₂ 填环约束；单独的满行秩选项则校正该 MLP 的仿射头以贴近源 teacher。两种路径都只有在**全部传入 TRAIN 点**经独立检查通过后才返回布局。目前这一有界路径仅支持一个所选简单环或 K4 型细分图的环族；普通 `fit` 不自动带此证书。两者均不保证新样本拓扑或类别完全分开。

## 结果

以下全部展示**训练中实际使用 PH 的 `DeepTDA`**，不是无 PH 的图方法。每张图**左格**是本项目，其余是外部实现。结构图画出**全部 TRAIN 点**及相同原始样本 ID 的源环边；行为图画出**全部 TEST 点**，拟合后才按类别着色。每格坐标范围完整独立，没有按标签挑点或裁掉离群点。

### 同源环结构：本项目、UMAP 与 TopoAE++

作者提供的 **K4 全部 300 点**有三个仅从源数据选定的独立环。显式使用结构引导的 `DeepTDA` 在完整 300 点上，经独立检查保持**同一批样本 ID 的三个源环（秩 3/3）**，H₀ 合并误差 **.046 ≤ .05**。UMAP 和作者 TopoAE++ 适配器输入相同，但输出尺度、计算预算不同；TopoAE++ 即使也显示三条显著输出条带，也不等于证明保留*同一组*源环。颜色表示源行序，**不是类别**。

![K4全部300训练点：左为本项目 PH 训练的引导式 DeepTDA，中为外部 UMAP，右为 TopoAE++ 作者核心CPU适配运行](assets/ph-guided-k4.png)

**COIL20-1 与下方 20 类 COIL-20 测试实验不是一个协议**：这里是**同一个真实物体的全部 72 个视角**，源数据无标签选出一个环。结构引导训练后的 `DeepTDA` 在全部72点上保持该同 ID 源环（**秩 1/1**），全域 H₀ 误差 **.046 ≤ .05**。颜色只是原始行序，**不是经核验的物理角度或类别**。两张都是传导式 **TRAIN 域**验收，不是新视角或牛活动的结果。

![COIL20-1全部72训练视角：左为本项目 PH 训练的引导式 DeepTDA，中为外部 UMAP，右为 TopoAE++ 作者核心CPU适配运行](assets/ph-guided-coil72.png)

UMAP 在 K4 和 COIL20-1 上的 overlap@15 分别为 **.868／.848**，高于本项目的 **.821／.816**；拓扑与局部邻居质量是两项不同指标。COIL20-1 的**行序**双邻居召回率为本项目 1.000、UMAP .861、TopoAE++ .972，**不能冒充已验证的物理角度召回率**。三者均为固定 seed0、计算预算不等；作者核心 CPU 适配结果不是论文的十次择优图。同源环与 H₀ 证书仅覆盖本项目声明的训练输入单位/样本 ID，不保证全部 H₁、链同构或新查询。**这些图没有单独证明采样 H₁ 损失的因果贡献**：额外加入的源结构引导布局和全域 H₀ 连接约束也是方法的一部分。[审查后的标量记录与限制](benchmarks/results/ph_guided_selected_cycles.json)。

### 行为与物体类别聚类：外部对照

#### Fashion-MNIST：PH 正则化降维与 UMAP

相同的仅由训练集拟合的 PCA64 输入；60,000 行拟合，10,000 行测试。固定 seed-0 图：

![Fashion-MNIST 全部测试点：左为 PH 正则化 DeepTDA，右为外部 UMAP](assets/phre-vs-external-fashion.png)

#### HAR：PH 正则化降维与 UMAP

官方受试者不重叠划分：7,352 行拟合、2,947 行测试；相同训练定义的数值参考：

![HAR 全部测试点：左为 PH 正则化 DeepTDA，右为外部 UMAP](assets/phre-vs-external-har.png)

#### COIL-20：PH 正则化降维、UMAP 与 TopoAE++

960 行拟合、480 行已见物体的留出视角，三种方法使用**完全相同顺序**的 PCA64 输入。TopoAE++ 是固定 seed-0 的**作者模型与损失核心＋独立 CPU 训练适配器**（1,000 次更新），不是原封不动的论文流水线或十次择优图。架构和计算预算不相同。

![COIL-20 全部测试视角：左为 PH 正则化 DeepTDA，中为外部 UMAP，右为 TopoAE++ 作者核心适配运行](assets/phre-vs-external-coil.png)

每种方法只在自身 TRAIN 布局上拟合 KMeans，再在**全部 TEST 行**评价；簇数对应数据集的类别数。这些 TEST 数据在项目开发过程中已经被查看，因此这里是**固定方法的描述性对照，而非完全未触碰的前瞻留出集**。同一数据集各方法输入一致，预先固定三个种子：

| 数据集 | PH 正则化降维（`DeepTDA`）TEST ARI | 外部 UMAP TEST ARI |
|---|---:|---:|
| Fashion-MNIST | .397 | **.471** |
| HAR | **.653** | .522 |
| COIL-20 | .391 | **.734** |

COIL-20 另有单独的 seed-0 三方对照：TRAIN 拟合的 **KMeans（20 簇、`n_init=10`）**得出 `DeepTDA` **.422**、UMAP **.735**、TopoAE++ 作者适配器 **.527**。这些**单种子数字不是**上表的三种子均值。ARI 衡量拟合后类别组织，而非同源环保持；上面的有界 TRAIN K4／COIL20-1 结果**没有改变**这里 Fashion、HAR、COIL-20 的聚类数字，也不等于通用拓扑保持。[聚类对照标量记录](benchmarks/results/phre_external_comparison.json)。

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
Z_new = model.transform(X_new)  # 新样本不自动继承拓扑证书
```

可显式请求**源环引导训练**，不改变普通 `fit` 的默认行为：

```python
model = DeepTDA(steps=200, h1_size=64, standardize=False,
                missing_indicators=False)
model.fit_with_topology_guidance(
    X_train, cycles=source_cycles, birth_radius=a, survival_radius=b,
    source_scale="median_all_pairs", h0_tolerance=0.05,
    strategy="single",  # 单环另可选 "single_beam"；三环可选 "subdivided_k4"
)
Z_train = model.embedding_   # 所有传入 TRAIN 点均已独立结构核验
Z_new = model.transform(X_new)  # 新总体需另行验证
```

对一个简单环，`strategy="single_beam"` 是**显式启用、有资源上限的源端候选搜索**，替代默认的贪心源布局。仅在此策略下可另设 `teacher_realization="affine_min_norm"`：对**同一个 PH 预训练 MLP**的仿射头插值源 teacher；默认仍是迭代的 `"adam"`。TRAIN 特征行不满秩、条件数过大或浮点更新不可表示时会拒绝。两种实现都须通过完整 TRAIN H₀/H₁ 独立验收；预算耗尽会报错，均不自动核验新点，也未隔离原生采样 H₁ 损失的因果作用。`source_cycles`、`a`、`b` 必须**只从源训练数据**在同一参考单位下选取，不得从标签或输出反选；一个源类可使用 [`auto_global_witnesses`](python/open_deep_tda/structural_auto_witness.py)，三类可使用 [`auto_source_h1_family`](python/open_deep_tda/structural_auto_family.py)，显式设 `allow_external=True`，并让可选的 Ripser 在**独立进程**运行。外部求解器有自己的资源限制。源结构不支持、预算耗尽或联合结构未通过时会明确失败，**不会返回未经核验的 embedding**。输入、源环和模型可能携带敏感信息，不要直接公开。普通默认几何 stress＋H₀/H₁/关键边是在有限**子云**上训练，并不等于全体数据的 PH。图片实际设置见[结构结果](benchmarks/results/ph_guided_selected_cycles.json)与[聚类结果](benchmarks/results/phre_external_comparison.json)；保存模型不等于匿名化。可运行 `python examples/run_ph_guided.py` 做**解析正确性示例**，它不是实数据效果基准。

## 署名与许可

[TopoAE++](https://github.com/MClemot/TopologicalAutoencodersPlusPlus) 和 [UMAP](https://github.com/lmcinnes/umap) 是独立的外部算法，不是本项目实现。标准 PH 和拓扑自编码器思想的署名见[第三方声明](THIRD_PARTY_NOTICES.md)。原创代码采用 [MIT](LICENSE)。
