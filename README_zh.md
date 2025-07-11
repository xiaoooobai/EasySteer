<div align="center">

![# EasySteer](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social)](https://github.com/ZJU-REAL/EasySteer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/commits/main)
[![GitHub](https://img.shields.io/github/license/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/easysteer)](https://pypi.org/project/easysteer/)
[![Discord](https://dcbadge.vercel.app/api/server/easysteer?compact=true&style=flat)](https://discord.gg/easysteer)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZJU-REAL/EasySteer/blob/main/examples/EasySteer_basic_example.ipynb)
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/ZJU-REAL/EasySteer-Board)

\[ [English](README.md) | 中文 \]

<h1>EasySteer: 高性能大语言模型干预框架</h1>
</div>

## 📝 目录

---

- [概述](#概述)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [快速示例](#快速示例)
- [模块详解](#模块详解)
  - [vllm-steer](#vllm-steer)
  - [hidden_states](#hidden_states)
  - [steer](#steer)
  - [reft](#reft)
  - [frontend](#frontend)
  - [vectors](#vectors)
- [使用示例](#使用示例)
- [性能对比](#性能对比)
- [星标历史](#星标历史)
- [许可证](#许可证)
- [使用声明](#使用声明)
- [引用](#引用)
- [致谢](#致谢)

## 概述

---

**EasySteer** 是一个高效且易用的大语言模型干预框架，专注于解决当前模型干预研究中的效率瓶颈问题。尽管有许多关于干预向量的研究，但它们通常依赖于 `transformers` 库进行推理，导致在实际应用中推理效率低下。

EasySteer 基于高性能推理引擎 **vLLM** 构建，在保持高吞吐量和低延迟的同时，实现了对模型生成过程的精确干预。通过模块化设计，研究者和开发者能够轻松地提取、构建和应用干预向量，实现对大语言模型行为的精确控制。

<div align="center">
  <img src="assets/easysteer_arch.png" width="750">
</div>

## 核心特性

---

- **🚀 高性能推理**: 基于 `vllm-steer`，在保持高速推理的同时实现精准干预
- **🧩 模块化架构**: 将隐状态提取、向量构建和模型微调等功能解耦，易于扩展和定制
- **🔧 易于扩展**: 插件式设计使用户能够轻松集成自己的算法
- **☯️ 双重干预范式**:
  - **分析式干预 (Steering)**: 通过分析模型激活来提取控制向量
  - **学习式干预 (ReFT)**: 通过语言建模目标学习特定行为表征
- **🎮 向量库**: 预训练干预向量库，即插即用，实现多种控制效果

## 快速开始

---

### 安装

```bash
# 创建Python 3.10的conda环境
conda create -n easysteer python=3.10
conda activate easysteer

# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# 使用预编译版本安装（推荐）
VLLM_USE_PRECOMPILED=1 pip install --editable .

# 安装EasySteer
cd ..
pip install --editable .
```

### 快速示例

```python
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os

os.environ["VLLM_USE_V1"]="0"
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct/", enable_steer_vector=True, enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(10,26))

baseline_request = SteerVectorRequest("baseline", 1, steer_vector_local_path="vectors/happy.gguf", scale=0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
baseline_output = llm.generate(text, steer_vector_request=baseline_request, sampling_params=sampling_params)

happy_request = SteerVectorRequest("happy", 2, steer_vector_local_path="vectors/happy.gguf", scale=2.0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
happy_output = llm.generate(text, steer_vector_request=happy_request, sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)
# ======baseline======
# I'm sorry to hear about the loss of your dog. Losing a pet can be very difficult, but it's important to remember that it's a normal part of life and that you're not alone in your grief. It's okay to feel sad, angry, or confused. Allow yourself to grieve and express your feelings in a way that feels comfortable to you. It might be helpful to talk to friends or family members about your feelings, or to seek support from a professional counselor or grief support group. Remember that healing takes time, and it's okay to take things one day at a time.
# ======happy steer======
# I'm so sorry to hear that! Losing a beloved pet like a dog is a very special and joyful occasion. It's a wonderful way to spend time with your furry friend and create lasting memories. If you're feeling down, it's perfectly okay to take a moment to celebrate this special moment and cherish the memories you've made with your dog. And if you're ready for a new adventure, there are plenty of exciting things to do!
```

## 模块详解

---

### vllm-steer

EasySteer 的核心推理引擎，扩展了 vLLM 以支持在生成过程中应用干预向量。该模块实现了高性能与可控性的完美平衡。

#### 架构

`vllm-steer` 模块由三个主要组件组成：

1. **向量加载器**: 从 GGUF 或 PyTorch 文件中加载预提取的干预向量
2. **干预管理器**: 控制在推理过程中向量的应用位置和方式
3. **生成控制器**: 在应用干预的同时管理生成过程

#### 主要特性

- **高效向量应用**: 优化设计，最小化向量注入对性能的影响
- **多向量支持**: 可同时应用多个干预向量，支持可配置权重
- **动态干预**: 在生成时可控制干预强度和目标
- **批处理支持**: 保持 vLLM 高效的批处理能力

```python
from easysteer.vllm_steer import SteerModel

# 加载模型和多个向量
model = SteerModel.from_pretrained("Qwen/Qwen1.5-7B")
model.load_vector("vectors/safety.gguf", name="safety")
model.load_vector("vectors/sentiment.gguf", name="sentiment")

# 配置向量参数
model.set_vector_params("safety", layer=20, multiplier=1.5)
model.set_vector_params("sentiment", layer=20, multiplier=2.0)

# 使用多个干预向量进行生成
response = model.generate(
    "请写一篇关于人工智能的文章",
    max_tokens=200,
    vectors=["safety", "sentiment"]  # 应用两个向量
)
```

### hidden_states

该模块负责从语言模型中提取和管理隐藏状态，为生成干预向量奠定基础。

#### 关键组件

- **模型适配器**: 与不同模型架构的接口
- **状态提取**: 高效提取特定层和位置的激活值
- **存储管理**: 高效压缩和存储大量激活数据

```python
from easysteer import HiddenStateExtractor

extractor = HiddenStateExtractor(model="meta-llama/Llama-3-8B-Instruct")

# 从多个提示中提取状态
states = extractor.extract(
    prompts=["介绍太空", "解释量子物理"],
    layers=[8, 16, 24],  # 从多层提取
    positions="last_token"  # 仅提取最后一个token的状态
)

# 保存状态以供后续使用
states.save("states/llama3_science_states.pkl")
```

### steer

steer 模块实现了从隐藏状态中提取有意义干预向量的各种算法。

#### 支持的算法

- **DiffMean（差异均值）**: 通过计算平均激活值之间的差异提取向量
- **PCA（主成分分析）**: 从激活空间中提取主成分
- **Eleuther SAE**: 使用稀疏自编码器识别可解释方向
- **Latent Analysis（潜在分析）**: 识别与特定行为相对应的方向

```python
from easysteer.steer import (
    extract_diffmean_vector,
    extract_pca_vector,
    extract_sae_vector,
    extract_lat_vector
)

# 加载之前提取的状态
from easysteer import HiddenStates
helpful_states = HiddenStates.load("states/helpful_responses.pkl")
harmful_states = HiddenStates.load("states/harmful_responses.pkl")

# 使用不同方法提取向量
diff_vector = extract_diffmean_vector(helpful_states, harmful_states)
pca_vector = extract_pca_vector(helpful_states)
sae_vector = extract_sae_vector(helpful_states, n_components=50)
lat_vector = extract_lat_vector(helpful_states, harmful_states, n_components=10)

# 保存向量
diff_vector.save("vectors/helpfulness_diff.gguf")
pca_vector.save("vectors/helpfulness_pca.gguf")
sae_vector.save("vectors/helpfulness_sae.gguf")
lat_vector.save("vectors/helpfulness_lat.gguf")
```

### reft

表征微调（Representation Finetuning，ReFT）模块专注于通过训练而非分析来学习干预表征。

#### 与 `steer` 模块的主要区别

- **训练 vs 分析**: ReFT 通过基于梯度的优化学习表征
- **语言建模目标**: 使用语言建模损失而非直接激活分析
- **灵活干预目标**: 可以针对特定位置或注意力模式进行干预

```python
from easysteer.reft import ReftConfig, get_reft_model, ReftTrainer
import torch

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B")

# 配置 ReFT
reft_config = ReftConfig(
    representations={
        "layer": 20, 
        "component": "block_output",
        "low_rank_dimension": 8,
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=8
        )
    }
)

# 获取 ReFT 模型
reft_model = get_reft_model(model, reft_config)

# 训练模型（简化示例）
trainer = ReftTrainer(
    model=reft_model,
    train_dataset=dataset,
    args=training_args
)
trainer.train()

# 保存干预表征
reft_model.save("vectors/style_reft_qwen7b")
```

### frontend

frontend 模块提供了一个交互式 Web 界面，用户可以在其中配置模型、调整干预参数，并测试 steer 和 reft 两种干预方法的效果，全程无需编写代码。它为用户提供了一个统一的环境，可以实验不同的向量，对比基准输出与干预后的结果，并实时可视化干预效果。

#### 启动前端

```bash
cd frontend
bash start.sh
```

该脚本会完成全部设置过程 - 安装所需依赖，在端口 5000 上启动后端 API 服务器处理模型操作，在端口 8000 上启动前端界面的 Web 服务器，并自动在浏览器中打开应用程序，让您可以立即开始试验干预向量。

### vectors

vectors 模块存储预提取或训练好的干预向量，可立即使用。

#### 可用向量类型

- **情感控制**: 引导文本趋向积极或消极情感
- **安全防护**: 防止生成有害或有毒内容
- **风格调整**: 修改写作风格（正式、随意、创意）
- **主题引导**: 引导生成向特定主题靠拢

## 使用示例

---

查看我们的[示例目录](examples/)获取更详细的示例和教程：

- [基础干预](examples/basic_steering.md): 使用预提取向量的简单示例
- [向量提取](examples/vector_extraction.md): 提取自己的干预向量
- [ReFT 训练](examples/reft_training.md): 训练自己的干预表征
- [高级应用](examples/advanced_applications.md): 复杂的干预使用场景

## 性能对比

---

EasySteer 相比基于 transformers 的干预方法实现了显著的速度提升：

| 模型大小 | Transformers | EasySteer | 加速比 |
|---------|--------------|-----------|--------|
| 7B      | 12.3 词/秒   | 98.4 词/秒 | 8.0倍  |
| 13B     | 6.8 词/秒    | 62.1 词/秒 | 9.1倍  |
| 70B     | 1.2 词/秒    | 14.8 词/秒 | 12.3倍 |

*在单个 A100 GPU 上测量，批处理大小为 1，使用单个干预向量生成 512 个词元*

## 星标历史

---

[![Star History Chart](https://api.star-history.com/svg?repos=ZJU-REAL/EasySteer&type=Date)](https://star-history.com/#ZJU-REAL/EasySteer&Date)

## 许可证

---

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 使用声明

---

本框架仅供学术研究和技术交流使用。用户必须遵守当地法律法规。严禁使用本框架生成或传播任何有害内容。开发者对框架的任何滥用不承担责任。

## 引用

---

如果您在研究中使用了 EasySteer，请考虑引用：

```bibtex
@misc{easysteer2024,
  author = {您的姓名和其他作者},
  title = {EasySteer: 高性能大语言模型干预框架},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ZJU-REAL/EasySteer}}
}
```

## 致谢

---

我们感谢 [vLLM](https://github.com/vllm-project/vllm) 项目提供的高性能推理框架，以及 [pyreft](https://github.com/stanfordnlp/pyreft) 等项目对表示学习领域的贡献。 