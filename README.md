<div align="center">

<img src="img/loge-1.png" width="160" alt="TCM-NER Logo"/>

# TCM-NER: 基于 GuwenBERT 与指针网络的中医古籍命名实体识别

**面向中医药古籍文献的高性能命名实体抽取模型（Named Entity Recognition for Ancient TCM Texts）**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-GuwenBERT-yellow.svg)](https://huggingface.co/ethanyt/guwenbert-base)
[![Task](https://img.shields.io/badge/Task-NER%20%7C%20Information%20Extraction-brightgreen.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Traditional%20Chinese%20Medicine-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English Documentation](README_EN.md) | [中文说明文档](README.md)

</div>

---

## 📖 项目简介

中医药古籍文献凝结了中华民族千百年来的医疗防病经验与理论精华。随着数字人文（Digital Humanities）与中医药信息学研究的深入推进，如何从海量非结构化的古籍医案与典籍文献中实现**精准实体识别与知识抽取**，已成为构建中医药知识图谱、支撑辅助诊疗推演的关键技术底座。

然而，中医古籍文本具有**字句精炼晦涩、文白词义演变显著、专有名词偏长、实体嵌套重叠**以及**篇章强话题性**等鲜明特点。传统基于 BIO/BMES 标签序列与条件随机场（CRF）的标注方案在面对长跨度实体、重叠实体及模糊边界时往往受限于标签转移约束与解码瓶颈。

针对上述挑战，**TCM-NER** 采用专为古代文言文预训练的古汉语语言模型（**GuwenBERT**）作为深层语义编码器，融合**双向指针网络（Pointer Network）**设计，提出实体边界“半指针-半标注”预测架构：
1. **语义理解**：借助预训练 GuwenBERT 深度提取古籍文言词法与上下文深层语义；
2. **边界解耦**：将实体抽取解耦为首尾边界识别任务，通过双路分类器分别打分预测实体的起始点（Start Pointer）与终止点（End Pointer）；
3. **重叠与多标签支持**：预测层采用独立 Sigmoid 概率激活函数，天然适配重叠实体、嵌套实体以及多义实体抽取场景。

---

## ✨ 核心特性

- 🏛️ **文言专有语义编码**：底座采用面向古代汉语大规模语料预训练的 `ethanyt/guwenbert-base` 模型，显著强化对古籍通假字、古今异义词及医籍专有术语的表征能力。
- 🎯 **双向指针边界解耦**：摒弃传统序列标注对 BIO 标签转移矩阵的强约束，将实体跨度抽取转化为起始位置与终止位置的并行分类预测，有效克服长实体断裂与组合爆炸瓶颈。
- 🔀 **重叠与嵌套实体抽取**：输出层由单标签互斥的 Softmax 改为独立 Sigmoid 激活，允许实体在同一文本区间内存在重叠与交叉标注。
- 📚 **经典医案古籍语料**：精选整理《朱丹溪医案》《叶天士医案》《薛立斋医案》《金匮方百家医案评议》等 20 部代表性中医药经典医案，覆盖药名、方剂、身体部位、年龄等多维核心实体类别。
- 🛠️ **完整工程链路支持**：提供从标注数据格式转换、动态 Padding 批处理数据加载、模型分布式训练（DDP）到细粒度 Bad Case 误差分析的一站式脚本。

---

## 📊 数据集与实体标注体系

### 1. 实体分类规范

本项目构建的中医古籍文本标注语料库，聚焦临床诊疗与理论推演的核心要素：

| 实体类别 | 英文标识 | 类别描述 | 古籍文本示例 |
| :--- | :--- | :--- | :--- |
| **中药名称** | `DRUG` | 单味中药材、饮片名、炮制品 | 白术、黄连、陈皮、青葱、生姜、砂糖、鹿首根 |
| **方剂名称** | `FORMULA` | 经方、时方、成方及临床经验方 | 平胃散、牛黄清心凉膈丸、活幼方 |
| **身体部位** | `BODY` | 人体脏腑、经络、官窍及解剖部位 | 心、舌、胃肠、肺、大肠、两颊 |
| **年龄特征** | `AGE` | 患者就诊时的年龄或时序生理特征 | 七十、小儿、壮年 |

### 2. 标注示例与展示

语料采用专业开源标注平台 Doccano 进行多阶段协同标注与质检清洗，代表性标注场景如下：

- **脏腑与官窍部位标注场景**：  
  ![figure 1](img/Picture1.png)  
  *示例：“心者，在天为热，在地为火，在人为心【身体部位】，在时为夏……通窍于舌【身体部位】。”*

- **方剂与病案处置标注场景**：  
  ![figure 2](img/Picture2.png)  
  *示例：“诸药不愈，用活幼方中牛黄清心凉膈丸【方剂】，敷服而愈。”*

---

## 🔍 中医古籍语言特性剖析

在实际文本挖掘与标注研究中，中医古籍呈现出以下深具挑战性的语言学与领域特性：

1. **篇章高度话题性（Topical Discourse Cohesion）**  
   古籍医案通常以特定病案或证候为核心展开系统论述，行文具备天然的话题聚合性。医家通常先记录患者主诉、体征脉象，随后展开辨证推演与病机论述，最终确立治则治法与遣药组方。在局部篇章中，证候词汇与方药实体高度共现并紧密关联。

2. **中医理论与现代医学的概念异构（Domain Semantic Discrepancy）**  
   中医术语体系根植于传统哲理与动态辨证观，与现代病理学词义存在深层差异：
   - **“病”与“证”的辨析**：现代医学的“病”通常指具有明确解剖与病理变化的独立疾病单位；而中医体系中判定疾病的标准是证候中最突出的综合症候群，从病因发端到转归的全局演进过程称之为“病”。
   - **多型共存与辨证论治**：以“疟疾”为例，中医在辨证视角下会进一步分为“正疟、温疟、寒疟、瘴疟（冷瘴/热瘴）、劳疟、疟母”等诸多不同证型，其病机病因大相径庭。因此，模型必须具备深层上下文理解能力，方能在复杂的文言语境下准确锚定实体边界与语义。

---

## 🏗️ 模型架构

本项目整体结构由**文言预训练编码层**与**双指针分类解码层**两部分构成：

![figure 3](img/Picture3.png)

### 1. 文本编码层（GuwenBERT Encoder）
对于输入的文言字序列 $X = (x_1, x_2, \dots, x_n)$，经由古汉语预训练语言模型（Transformer 架构）提取深层上下文双向表征：
$$H = \text{GuwenBERT}(X) \in \mathbb{R}^{n \times d}$$
其中 $d = 768$ 为模型隐藏层维度。

### 2. 双指针边界预测层（Pointer Classification Layer）
为了有效预测任意长度实体的边界并支持重叠抽取，模型设计了起始位置与终止位置两个平行的分类头：
- **起始位置分类器（Start Pointer）**：
  $$P_{start} = \sigma(W_{start} H + b_{start}) \in \mathbb{R}^{n \times 1}$$
- **终止位置分类器（End Pointer）**：
  $$P_{end} = \sigma(W_{end} H + b_{end}) \in \mathbb{R}^{n \times 1}$$

### 3. 解码与匹配策略（Span Decoding）
在推理阶段，通过预设置信度阈值（如 $P_{start} \ge 0.7, P_{end} \ge 0.6$）分别筛选出合法的起始索引集 $\mathcal{S}$ 与终止索引集 $\mathcal{E}$。针对每一个起始索引 $i \in \mathcal{S}$，在终止候选集 $\mathcal{E}$ 中检索满足 $j \ge i$ 的终止边界，并截取对应的文本片段 $[i, j]$ 作为候选实体输出。

### 4. 损失函数（Loss Function）
模型采用双路多标签二值交叉熵（Binary Cross-Entropy Loss）联合优化：
$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{BCE}(P_{start}, Y_{start}) + \mathcal{L}_{BCE}(P_{end}, Y_{end}) \right)$$

---

## 📁 目录结构

```text
.
├── DataFormat/                  # 数据处理工具与格式转换脚本
│   ├── Format.py                # 文本清洗与格式预处理
│   ├── NER_Format.py            # Doccano 原始标注转换为模型标准格式
│   └── Split_File.py            # 数据集切分脚本
├── data/                        # 数据集存放目录
│   └── 4-3.zip                  # 原始标注压缩数据
├── img/                         # 架构图与说明配图
│   ├── Picture1.png             # 实体标注示例图 1（身体部位）
│   ├── Picture2.png             # 实体标注示例图 2（方剂）
│   ├── Picture3.png             # 模型架构原理图
│   └── loge-1.png               # 项目 Logo 标志
├── dataloder.py                 # 数据加载器与动态 Padding 模块
├── main.py                      # 模型构建、训练循环与综合评估主入口
├── test_Modules.py              # 辅助组件测试脚本
├── test_generator.py            # 数据生成器测试脚本
├── Example of data labeling.txt # 详细的数据标注样例集合
├── README.md                    # 中文说明文档
└── README_EN.md                 # 英文说明文档 (English Documentation)
```

---

## 🚀 快速开始

### 1. 环境准备

推荐使用 Python 3.8+ 及 CUDA 11.0+ 环境：

```bash
git clone https://github.com/KuoChing-cmd/ZhongYi-NER2.git
cd ZhongYi-NER2
pip install torch torchvision transformers numpy tqdm
```

### 2. 预训练模型权重准备

模型依托古汉语文言文预训练模型 `ethanyt/guwenbert-base`。可在项目根目录下创建 `guwenbert-base` 文件夹并下载权重，或直接通过 HuggingFace Hub 加载：

```bash
# 从 HuggingFace Hub 下载预训练权重（可选）
git lfs install
git clone https://huggingface.co/ethanyt/guwenbert-base
```

### 3. 数据集格式与预处理

本项目标准输入格式为包含 `text` 和 `ner` 列表的 JSON 结构，样例文件可见 [`Example of data labeling.txt`](Example%20of%20data%20labeling.txt)：

```json
[
  {
    "text": "族叔祖年七十，禀甚壮，形甚瘦。夏末患泄利至深秋，百方不应……以荣萸、陈皮、青葱、鹿首根、生姜煎浓汤，和以砂糖，饮一碗许……又与平胃散加白术、黄连，旬日而安。",
    "ner": [
      "七十",
      "胃肠",
      "肺",
      "大肠",
      "陈皮",
      "青葱",
      "生姜",
      "砂糖",
      "平胃散",
      "白术",
      "黄连"
    ]
  }
]
```

若使用 Doccano 导出的原始标注数据，可运行转换脚本进行一键清洗格式化：

```bash
python DataFormat/NER_Format.py
```

### 4. 模型训练

执行主训练程序：

```bash
python main.py
```

训练过程将基于分布式训练架构进行前向传播与反向优化，并在验证集上自动评测，保存效果最佳的模型权重文件 `NER_best.pth`。

### 5. 推理抽取示例

模型训练完成后，可运行如下推理函数进行古籍文本实体识别：

```python
import torch
import numpy as np
from transformers import BertTokenizer

# 初始化 Tokenizer 与加载模型
tokenizer = BertTokenizer.from_pretrained("ethanyt/guwenbert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("NER_best.pth", map_location=device)
model.eval()

def extract_entities(text: str, start_threshold=0.7, end_threshold=0.6):
    token_ids = tokenizer.encode(text)
    inputs = torch.as_tensor([token_ids], device=device)
    
    with torch.no_grad():
        pn1, pn2 = model(inputs)
        pn1 = pn1.squeeze(0).squeeze(-1).cpu().numpy()
        pn2 = pn2.squeeze(0).squeeze(-1).cpu().numpy()
        
    start_indices = np.where(pn1 > start_threshold)[0]
    end_indices = np.where(pn2 > end_threshold)[0]
    
    entities = set()
    for s in start_indices:
        e_candidates = end_indices[end_indices >= s]
        for e in e_candidates:
            span = text[s:e]
            if span:
                entities.add(span)
    return entities

# 示例验证
sample_text = "心者，在天为热，在地为火，在人为心，在时为夏，在方为志，在色为赤，通窍于舌。"
print("抽取结果：", extract_entities(sample_text))
```

---

## 📈 评测指标与误差分析

主评估脚本在验证集上自动计算标准的序列抽取评测指标（**Precision**、**Recall** 及 **F1-Score**）：

$$Precision = \frac{|R \cap T|}{|R|}, \quad Recall = \frac{|R \cap T|}{|T|}, \quad F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

其中 $R$ 为模型预测实体集合，$T$ 为真实标注实体集合。评估详情将保存于 `./data/eval.csv`，记录内容包括：
- `text`：原始古籍医案文本
- `ner_list`：真实标准实体列表
- `ner_list_pred`：模型预测抽取的实体列表
- `new`：模型预测但未在标注中的新增实体（用于挖掘潜在漏标）
- `lack`：人工标注但模型遗漏的实体（用于分析难点案例）

---

## ⚖️ 免责声明

1. 本项目所呈现的实验数据与模型表现均基于特定古籍文献语料和实验超参数配置，并不代表模型在全量古籍场景下的绝对性能。
2. 实验结果可能因随机数种子设置、硬件运算环境及软件依赖版本不同而存在合理差异。
3. 本项目所有代码、数据处理逻辑及模型实现仅供**学术研究、数字人文探索及自然语言处理技术交流**使用，**严禁用于实际临床诊断或作为任何现实医疗决策的依据**。
4. 使用者须在开源许可协议范围内合规使用本项目成果，开发团队对因使用本项目内容而引起的任何直接或间接后果不承担法律责任。

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 许可协议开源。
