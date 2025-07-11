<div align="center">

![# EasySteer](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social)](https://github.com/ZJU-REAL/EasySteer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/commits/main)
[![GitHub](https://img.shields.io/github/license/ZJU-REAL/EasySteer)](https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/easysteer)](https://pypi.org/project/easysteer/)
[![Discord](https://dcbadge.vercel.app/api/server/easysteer?compact=true&style=flat)](https://discord.gg/easysteer)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZJU-REAL/EasySteer/blob/main/examples/EasySteer_basic_example.ipynb)
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/ZJU-REAL/EasySteer-Board)

\[ English | [中文](README_zh.md) \]

<h1>EasySteer: High-Performance LLM Steering Framework</h1>
</div>

## 📝 Table of Contents

---

- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quick Example](#quick-example)
- [Modules](#modules)
  - [vllm-steer](#vllm-steer)
  - [hidden_states](#hidden_states)
  - [steer](#steer)
  - [reft](#reft)
  - [frontend](#frontend)
  - [vectors](#vectors)
- [Examples](#examples)
- [Performance](#performance)
- [Star History](#star-history)
- [License](#license)
- [Usage Statement](#usage-statement)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

---

**EasySteer** is an efficient and easy-to-use framework for steering large language models, focusing on solving efficiency bottlenecks in current model intervention research. While numerous studies on steering vectors exist, they typically rely on the `transformers` library for inference, resulting in low efficiency in real-world applications.

Built on the high-performance inference engine **vLLM**, EasySteer achieves precise interventions during model generation while maintaining high throughput and low latency. Through its modular design, researchers and developers can easily extract, construct, and apply steering vectors to achieve fine-grained control over LLM behavior.

<div align="center">
  <img src="assets/easysteer_arch.png" width="750">
</div>

## Key Features

---

- **🚀 High-Performance Inference**: Based on `vllm-steer`, achieving precise interventions while maintaining fast inference speeds
- **🧩 Modular Architecture**: Decoupled hidden state extraction, vector construction, and model fine-tuning for easy expansion and customization
- **🔧 Easy Extension**: Plugin-based design allowing users to easily integrate their own algorithms
- **☯️ Dual Intervention Paradigms**:
  - **Analysis-based Intervention (Steering)**: Extract control vectors by analyzing model activations
  - **Learning-based Intervention (ReFT)**: Learn specific behavioral representations through language modeling objectives
- **🎮 Vector Library**: Pre-extracted intervention vectors ready for use with various control effects

## Getting Started

---

### Installation

```bash
# Create conda environment with Python 3.10
conda create -n easysteer python=3.10
conda activate easysteer

# Clone the repository (with submodules)
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# Install with pre-compiled version (recommended)
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Install EasySteer
cd ..
pip install --editable .
```

### Quick Example

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

## Modules

---

### vllm-steer

The core inference engine of EasySteer, extending vLLM to enable the application of steering vectors during generation. This module achieves the perfect balance between high performance and controllability.

#### Architecture

The `vllm-steer` module consists of three main components:

1. **Vector Loader**: Loads pre-extracted steering vectors from GGUF or PyTorch files
2. **Intervention Manager**: Controls where and how vectors are applied during inference
3. **Generation Controller**: Manages the generation process while applying interventions

#### Key Features

- **Efficient Vector Application**: Optimized to minimize the performance impact of vector injection
- **Multi-Vector Support**: Apply multiple steering vectors simultaneously with configurable weights
- **Dynamic Intervention**: Control intervention strength and targeting at generation time
- **Batch Processing**: Maintain vLLM's efficient batch processing capabilities

```python
from easysteer.vllm_steer import SteerModel

# Load model and multiple vectors
model = SteerModel.from_pretrained("Qwen/Qwen1.5-7B")
model.load_vector("vectors/safety.gguf", name="safety")
model.load_vector("vectors/sentiment.gguf", name="sentiment")

# Configure vector parameters
model.set_vector_params("safety", layer=20, multiplier=1.5)
model.set_vector_params("sentiment", layer=20, multiplier=2.0)

# Generate with multiple steering vectors
response = model.generate(
    "Write about artificial intelligence",
    max_tokens=200,
    vectors=["safety", "sentiment"]  # Apply both vectors
)
```

### hidden_states

This module extracts and manages hidden states from language models, forming the foundation for steering vector generation.

#### Key Components

- **Model Adapters**: Interface with different model architectures
- **State Extraction**: Efficient extraction of activations at specific layers and positions
- **Storage Management**: Compress and store large activation datasets efficiently

```python
from easysteer import HiddenStateExtractor

extractor = HiddenStateExtractor(model="meta-llama/Llama-3-8B-Instruct")

# Extract states from multiple prompts
states = extractor.extract(
    prompts=["Tell me about space", "Explain quantum physics"],
    layers=[8, 16, 24],  # Extract from multiple layers
    positions="last_token"  # Extract only last token states
)

# Save the states for later use
states.save("states/llama3_science_states.pkl")
```

### steer

The steer module implements various algorithms for extracting meaningful intervention vectors from hidden states.

#### Supported Algorithms

- **DiffMean**: Extract vectors by computing differences between mean activations
- **PCA**: Extract principal components from activation spaces
- **Eleuther SAE**: Use sparse autoencoders to identify interpretable directions
- **Latent Analysis**: Identify directions corresponding to specific behaviors

```python
from easysteer.steer import (
    extract_diffmean_vector,
    extract_pca_vector,
    extract_sae_vector,
    extract_lat_vector
)

# Load previously extracted states
from easysteer import HiddenStates
helpful_states = HiddenStates.load("states/helpful_responses.pkl")
harmful_states = HiddenStates.load("states/harmful_responses.pkl")

# Extract vectors using different methods
diff_vector = extract_diffmean_vector(helpful_states, harmful_states)
pca_vector = extract_pca_vector(helpful_states)
sae_vector = extract_sae_vector(helpful_states, n_components=50)
lat_vector = extract_lat_vector(helpful_states, harmful_states, n_components=10)

# Save the vectors
diff_vector.save("vectors/helpfulness_diff.gguf")
pca_vector.save("vectors/helpfulness_pca.gguf")
sae_vector.save("vectors/helpfulness_sae.gguf")
lat_vector.save("vectors/helpfulness_lat.gguf")
```

### reft

The Representation Finetuning (ReFT) module focuses on learning intervention representations through training rather than analytical extraction.

#### Key Differences from `steer`

- **Training vs Analysis**: ReFT learns representations through gradient-based optimization
- **Language Modeling Objective**: Uses language modeling loss rather than direct activation analysis
- **Flexible Intervention Targets**: Can target specific positions or attention patterns

```python
from easysteer.reft import ReftConfig, get_reft_model, ReftTrainer
import torch

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B")

# Configure ReFT
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

# Get ReFT model
reft_model = get_reft_model(model, reft_config)

# Train the model (simplified example)
trainer = ReftTrainer(
    model=reft_model,
    train_dataset=dataset,
    args=training_args
)
trainer.train()

# Save intervention representation
reft_model.save("vectors/style_reft_qwen7b")
```

### frontend

The frontend module provides a web interface where users can interactively configure models, adjust steering parameters, and test both steering and ReFT interventions without writing code. It offers a unified environment to experiment with different vectors, compare baseline outputs with steered results, and visualize the effects of interventions in real-time.

#### Starting the Frontend

```bash
cd frontend
bash start.sh
```

This script handles the complete setup process - it installs required dependencies, launches the backend API server on port 5000 to handle model operations, starts a web server on port 8000 for the frontend interface, and automatically opens your browser to the application where you can immediately begin experimenting with steering vectors.

### vectors

The vectors module stores pre-extracted or trained intervention vectors for immediate use.

#### Available Vector Types

- **Sentiment Control**: Steer text toward positive or negative sentiment
- **Safety Guardrails**: Prevent generation of harmful or toxic content
- **Style Adjustment**: Modify the writing style (formal, casual, creative)
- **Topic Guidance**: Steer generation toward specific topics

## Examples

---

Check out our [examples directory](examples/) for more detailed examples and tutorials:

- [Basic Steering](examples/basic_steering.md): Simple examples of using pre-extracted vectors
- [Vector Extraction](examples/vector_extraction.md): Extract your own steering vectors
- [ReFT Training](examples/reft_training.md): Train your own intervention representations
- [Advanced Applications](examples/advanced_applications.md): Complex steering use cases

## Performance

---

EasySteer achieves significant speedups compared to transformers-based steering approaches:

| Model Size | Transformers | EasySteer | Speedup |
|------------|--------------|-----------|---------|
| 7B         | 12.3 tok/s   | 98.4 tok/s | 8.0x    |
| 13B        | 6.8 tok/s    | 62.1 tok/s | 9.1x    |
| 70B        | 1.2 tok/s    | 14.8 tok/s | 12.3x   |

*Measured on a single A100 GPU, batch size 1, generating 512 tokens with a single steering vector*

## Star History

---

[![Star History Chart](https://api.star-history.com/svg?repos=ZJU-REAL/EasySteer&type=Date)](https://star-history.com/#ZJU-REAL/EasySteer&Date)

## License

---

This project is licensed under the [Apache License 2.0](LICENSE).

## Usage Statement

---

This framework is intended for academic research and technical exchange only. Users must comply with local laws and regulations. It is strictly prohibited to use this framework to generate or disseminate any harmful content. The developers are not responsible for any misuse of this framework.

## Citation

---

If you find EasySteer useful in your research, please consider citing:

```bibtex
@misc{easysteer2024,
  author = {Your Name and Other Authors},
  title = {EasySteer: A High-Performance Framework for LLM Steering},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ZJU-REAL/EasySteer}}
}
```

## Acknowledgements

---

We thank the [vLLM](https://github.com/vllm-project/vllm) project for providing the high-performance inference framework, and projects like [pyreft](https://github.com/stanfordnlp/pyreft) for their contributions to the field of representation learning. 