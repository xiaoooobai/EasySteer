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

**EasySteer** 是一个高效且易用的大语言模型干预框架，专注于解决当前模型干预研究中的效率瓶颈问题。尽管有许多关于干预向量的研究，但它们通常依赖于 `transformers` 库进行推理，导致在实际应用中推理效率低下。

EasySteer 基于高性能推理引擎 **vLLM** 构建，在保持高吞吐量和低延迟的同时，实现了对模型生成过程的精确干预。通过模块化设计，研究者和开发者能够轻松地提取、构建和应用干预向量，实现对大语言模型行为的精确控制。

<div align="center">
  <img src="assets/easysteer_arch.png" width="750">
</div>

## 核心特性

- **🚀 高性能推理**: 基于 `vllm-steer`，在保持高速推理的同时实现精准干预
- **🧩 模块化架构**: 将隐状态提取、向量构建和模型微调等功能解耦，易于扩展和定制
- **🔧 易于扩展**: 插件式设计使用户能够轻松集成自己的算法
- **☯️ 双重干预范式**:
  - **分析式干预 (Steering)**: 通过分析模型激活来提取控制向量
  - **学习式干预 (ReFT)**: 通过语言建模目标学习特定行为表征
- **🎮 向量库**: 预训练干预向量库，即插即用，实现多种控制效果

## 快速开始

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

# 设置使用vLLM v0版本，当前steer功能不支持v1版本
os.environ["VLLM_USE_V1"]="0"

# 初始化LLM模型
# enable_steer_vector=True: 启用向量干预功能（不设置则与原始vLLM相同）
# enforce_eager=True: 确保干预的可靠性和稳定性（强烈建议设置）
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

### vllm-steer

EasySteer 的核心推理引擎，扩展了 vLLM 以支持在生成过程中应用干预向量。该模块具有以下特性：

- **高性能向量应用**：利用了 vLLM 的高效推理能力
- **多向量复杂控制策略**：支持同时应用多个干预向量，实现复杂的组合干预效果
- **精准干预控制**：精确设定干预的目标位置、应用层级和干预强度
- **扩展接口设计**：提供标准化接口，使研究人员能轻松实现和集成自定义干预算法

#### 内部结构

`vllm-steer` 的核心功能在 `vllm/steer_vectors` 目录中实现，其文件结构组织如下：

```
vllm/steer_vectors/
├── __init__.py                # 模块入口
├── request.py                 # 请求和配置定义
├── models.py                  # 模型集成与向量注册
├── layers.py                  # 自定义层实现
├── worker_manager.py          # 工作线程管理
└── algorithms/                # 各类干预算法实现
    ├── __init__.py            # 算法注册
    ├── base.py                # 算法基类与接口定义
    ├── factory.py             # 算法工厂（用于创建算法实例）
    ├── direct.py              # 直接干预算法
    ├── loreft.py              # LoReFT算法实现
    ├── multi_vector.py        # 多向量组合算法
    └── template.py            # 新算法模板示例
```

#### 核心组件

1. **请求与配置系统** (`request.py`):
   - `SteerVectorRequest`: 定义干预向量请求格式，支持单向量和多向量模式
   - `VectorConfig`: 多向量模式下单个向量的配置定义

2. **算法框架** (`algorithms/base.py`):
   - `BaseSteerVectorAlgorithm`: 所有干预算法的抽象基类，定义标准接口
   - 提供位置解析、触发条件检查等通用功能

3. **算法工厂** (`algorithms/factory.py`):
   - 负责根据配置动态创建适当的算法实例
   - 支持算法注册机制，便于扩展新算法

4. **向量应用实现**:
   - `direct.py`: 实现直接向量干预（最基本的加法干预）
   - `loreft.py`: 实现LoReFT低秩适应的干预方法
   - `multi_vector.py`: 实现多向量组合干预策略

#### 扩展机制

`vllm-steer` 设计了灵活的扩展机制，使研究者可以轻松实现和集成自己的干预算法：

1. **基于接口的插件架构**:
   - 所有算法都继承自 `BaseSteerVectorAlgorithm` 基类
   - 通过实现标准接口方法添加新算法，无需修改框架核心代码

2. **算法注册系统**:
   - 在 `algorithms/__init__.py` 中注册新算法
   - 通过工厂模式自动加载和实例化算法

3. **模板示例**:
   - `template.py` 提供新算法开发模板，包含详细注释
   - 遵循模板开发可确保与框架无缝集成

4. **多层级干预点**:
   - 支持在模型不同层级（如注意力层、FFN层等）应用干预
   - 通过 `forward_decoder_layer` 和 `forward_mlp_layer` 等钩子实现

#### 扩展新算法示例

要添加新的干预算法，只需以下几步：

1. 创建新的算法类（继承 `BaseSteerVectorAlgorithm`）
2. 实现必要的接口方法（如 `load_from_path`, `apply_intervention` 等）
3. 在算法注册系统中注册新算法
4. 通过配置使用新算法

```python
# 示例：实现一个新的干预算法
from vllm.steer_vectors.algorithms.base import BaseSteerVectorAlgorithm
import torch

class MyCustomAlgorithm(BaseSteerVectorAlgorithm):
    """自定义干预算法实现"""
    
    @classmethod
    def load_from_path(cls, path, device, **kwargs):
        # 加载向量文件实现
        vector_data = torch.load(path, map_location=device)
        return {"vector": vector_data, "other_params": ...}
    
    def __init__(self, layer_id=None):
        super().__init__(layer_id)
        self.vector = None
        self.scale = 1.0
        
    def set_steer_vector(self, index, vector, scale=1.0, **kwargs):
        self.vector = vector
        self.scale = scale
    
    def apply_intervention(self, hidden_states):
        # 自定义干预逻辑
        if self.vector is not None:
            return hidden_states + self.scale * self.vector
        return hidden_states
    
    # 实现其他必要的接口方法...

# 在algorithms/__init__.py中注册:
# ALGORITHM_CLASSES["my_custom"] = MyCustomAlgorithm
```

通过这种模块化设计，研究人员可以专注于干预算法的核心逻辑实现，而无需了解底层推理引擎的复杂细节。

#### 向量配置示例

```python
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

# 示例1：单向量干预配置
single_vector_request = SteerVectorRequest(
    steer_vector_name="sentiment_control",       # 向量名称（用于日志和调试）
    steer_vector_id=1,                           # 向量ID（用于内部标识）
    steer_vector_local_path="vectors/happy.gguf",# 向量文件路径
    scale=2.0,                                   # 应用强度（正值增强，负值抑制）
    target_layers=[10, 11, 12],                  # 目标层（指定应用的模型层）
    prefill_trigger_tokens=[-1],                 # 预填充阶段要干预的token ID（-1表示全部token）
    generate_trigger_tokens=[-1]                 # 生成阶段要干预的token ID（-1表示全部token）
)

# 示例2：多向量干预配置
multi_vector_request = SteerVectorRequest(
    # 向量请求的基本信息
    steer_vector_name="multi_direction_control",  # 向量组合名称
    steer_vector_id=2,                            # 向量组合ID
    
    # 配置多个不同方向的干预向量
    vector_configs=[
        # 第一个向量配置
        VectorConfig(
            path="vector_direction1.gguf",         # 向量文件路径
            scale=1.5,                             # 正向强度（增强此方向）
            target_layers=[20],                    # 应用于模型第20层
            prefill_trigger_positions=[-2],        # 干预prompt中倒数第二个token位置
            algorithm="direct",                    # 应用算法
            normalize=False                        # 是否规范化向量
        ),
        
        # 第二个向量配置
        VectorConfig(
            path="vector_direction2.gguf",         # 向量文件路径
            scale=-0.8,                            # 负向强度（抑制此方向）
            target_layers=[20],                    # 应用于模型第20层
            prefill_trigger_positions=[-2],        # 干预prompt中倒数第二个token位置
            algorithm="direct",                    # 应用算法
            normalize=False                        # 是否规范化向量
        ),
        
        # 第三个向量配置
        VectorConfig(
            path="vector_direction3.gguf",         # 向量文件路径
            scale=-1.0,                            # 负向强度（抑制此方向）
            target_layers=[20],                    # 应用于模型第20层
            prefill_trigger_positions=[-2],        # 干预prompt中倒数第二个token位置
            algorithm="direct",                    # 应用算法 
            normalize=False                        # 是否规范化向量
        ),
    ],
    
    # 多向量干预的附加参数
    debug=False,                                   # 是否输出调试信息
    conflict_resolution="sequential"               # 冲突解决策略：按顺序应用
)
```

### hidden_states

该模块负责从语言模型中提取和管理隐藏状态，为生成干预向量奠定基础。

```python
# 导入hidden_states模块以提取模型激活值
import easysteer.hidden_states as hs

# 创建一个新的LLM实例，设置为reward模式
# 注意：这允许我们提取隐藏状态而不是生成文本
llm = LLM(
    model="path/to/your/model",  # 模型路径
    task="reward",               # 使用reward任务获取隐藏状态
    tensor_parallel_size=1
)

# 准备一些示例提示
prompts = [
    "人工智能的未来发展趋势是什么？",
    "请解释量子计算的基本原理",
    "如何有效地学习一门新语言"
]

# 提取所有token的隐藏状态
all_hidden_states, outputs = hs.get_all_hidden_states(llm, prompts)
```

### steer

steer 模块实现了从隐藏状态中提取有意义干预向量的各种算法，包括 DiffMean（差异均值）、PCA（主成分分析）、LAT（Linear artificial tomography，线性人工断层扫描）、Linear probe（线性探针）以及 SAE（稀疏自编码器）。这些算法各有优势，可以根据不同场景和需求进行选择。

```python
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector

# 使用差异均值方法提取控制向量
control_vector = extract_diffmean_control_vector(
    all_hidden_states=all_hidden_states,  # 三维列表 [样本][layer][token]
    positive_indices=[0, 1, 2, 3],     # 正样本索引
    negative_indices=[4, 5, 6, 7],     # 负样本索引
    model_type="qwen2.5",  
    token_pos=-1,      # 使用最后一个token（默认）
    normalize=True
)

# 导出控制向量为GGUF格式
control_vector.export_gguf("vectors/diffmean.gguf")

# 导入已保存的控制向量
control_vector = StatisticalControlVector.import_gguf("vectors/diffmean.gguf")
```

### reft

Steering属于分析式干预，通过分析提取到的hidden states来提取控制向量。而ReFT属于学习式干预，通过语言建模目标学习特定行为表征。本模块重构了pyreft项目。

```python
import torch
import transformers
import easysteer.reft as reft

# 加载原始语言模型
model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda"
)

# 获取tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# 设置ReFT配置，使用BiasIntervention
reft_config = reft.ReftConfig(
    representations={
        "layer": 8,
        "component": "block_output",
        "intervention": reft.BiasIntervention(
            embed_dim=model.config.hidden_size
        ),
    }
)

# 获取ReFT模型
reft_model = reft.get_reft_model(model, reft_config)

# 准备训练数据示例（提示和目标输出）
prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
training_examples = [
    ["Who are you?", "🤖💬🌐🧠"],
    ["What's 2+2?", "🔢➕🔢➡️4️⃣"],
    ["Why is the sky blue?", "🌍🛡️☀️➡️🔵🌌"],
    # ... 更多训练样例
]

# 创建数据模块
data_module = reft.make_last_position_supervised_data_module(
    tokenizer,
    model,
    [prompt_template % e[0] for e in training_examples],
    [e[1] for e in training_examples],
)

# 设置训练参数
training_args = transformers.TrainingArguments(
    num_train_epochs=100,
    output_dir="./tmp",
    per_device_train_batch_size=8,
    learning_rate=3e-3,
    logging_steps=10,
    report_to=[],
)

# 创建训练器并训练
trainer = reft.ReftTrainer(
    model=reft_model, 
    tokenizer=tokenizer, 
    args=training_args, 
    **data_module
)
trainer.train()

# 保存训练好的干预表征
reft_model.save("results/emoji_style")
```

### frontend

frontend 模块提供了一个交互式 Web 界面，用户可以在其中配置模型、调整干预参数，并测试 steer 和 reft 两种干预方法的效果，全程无需编写代码。它为用户提供了一个统一的环境，可以实验不同的向量，对比基准输出与干预后的结果，并实时可视化干预效果。

#### 启动前端

```bash
cd frontend
bash start.sh
```

## 使用示例

EasySteer 提供了两类资源帮助用户快速上手：

1. **examples** 文件夹包含多个简单使用示例
2. **replications** 文件夹包含使用 EasySteer 复现的学术论文实验

### 论文复现

下表列出了使用 EasySteer 复现的重要论文工作：

| 论文标题 | 分类 | 链接 |
|---------|------|-----|
| SEAL: Steerable Reasoning Calibration of Large Language Models for Free | thinking pattern | [复现代码](replications/seal/) |
| _更多复现持续添加中..._ | | |

## 性能对比

EasySteer 相比基于 transformers 的干预方法实现了显著的速度提升：

| 模型大小 | Transformers | EasySteer | 加速比 |
|---------|--------------|-----------|--------|
| 7B      | 12.3 词/秒   | 98.4 词/秒 | 8.0倍  |
| 13B     | 6.8 词/秒    | 62.1 词/秒 | 9.1倍  |
| 70B     | 1.2 词/秒    | 14.8 词/秒 | 12.3倍 |

*在单个 A100 GPU 上测量，批处理大小为 1，使用单个干预向量生成 512 个词元*

## 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=ZJU-REAL/EasySteer&type=Date)](https://star-history.com/#ZJU-REAL/EasySteer&Date)

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 使用声明

本框架仅供学术研究和技术交流使用。用户必须遵守当地法律法规。严禁使用本框架生成或传播任何有害内容。开发者对框架的任何滥用不承担责任。

## 引用

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

我们感谢 [vLLM](https://github.com/vllm-project/vllm) 项目提供的高性能推理框架，以及 [pyreft](https://github.com/stanfordnlp/pyreft) 等项目对表示学习领域的贡献。 