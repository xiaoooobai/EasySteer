# Steering Methods Package

统一的控制向量提取方法包，包含了常用的统计方法。

## 文件结构

```
steer_method/
├── __init__.py                    # 包初始化，导出所有接口
├── utils.py                       # 工具类和通用函数
├── diffmean.py                    # DiffMean方法
├── pca.py                         # PCA方法
├── lat.py                         # LAT方法
├── linear_probe.py                # LinearProbe方法
├── unified_interface.py           # 统一接口函数
└── README.md                      # 说明文档
```

## 支持的方法

### 1. DiffMean
- **描述**: 正负样本均值差
- **输入**: 正负样本的隐藏状态
- **用途**: 对比学习，概念区分

### 2. PCA
- **描述**: 主成分分析（默认只使用正样本）
- **输入**: 正样本的隐藏状态
- **用途**: 主要变化方向

### 3. LAT (Linear Algebraic Technique)
- **描述**: 对随机配对差值的PCA
- **输入**: 正样本的隐藏状态
- **用途**: 鲁棒的方向提取

### 4. LinearProbe
- **描述**: 线性探测器方法
- **输入**: 隐藏状态
- **用途**: 分类器权重，监督学习

## 快速开始

### 基本使用

```python
from EasySteerTest.steer_method import extract_diffmean_control_vector

# 提取DiffMean控制向量（使用最后一个token）
control_vector = extract_diffmean_control_vector(
    all_hidden_states=all_hidden_states,  # 三维列表 [样本][layer][token]
    positive_indices=[0, 1, 2, 3, 4],     # 正样本索引
    negative_indices=[5, 6, 7, 8, 9],     # 负样本索引
    model_type="llama-2-7b",
    token_pos=-1,      # 使用最后一个token（默认）
    normalize=True
)

# 保存为GGUF格式
control_vector.export_gguf("my_control_vector.gguf")
```

### 统一接口

```python
from EasySteerTest.steer_method import extract_statistical_control_vector

# 使用统一接口
control_vector = extract_statistical_control_vector(
    method="pca",  # 或 "diffmean", "lat", "linear_probe" 等
    all_hidden_states=all_hidden_states,
    positive_indices=positive_indices,
    model_type="llama-2-7b",
    token_pos="last"  # 可选：指定token位置
)
```

### 加载已保存的控制向量

```python
from EasySteerTest.steer_method import StatisticalControlVector

# 从GGUF文件加载
control_vector = StatisticalControlVector.import_gguf("my_control_vector.gguf")

print(f"Method: {control_vector.method}")
print(f"Layers: {list(control_vector.directions.keys())}")
print(f"Metadata: {control_vector.metadata}")
```

## Token位置选择

大多数方法支持指定使用哪个token的hidden state进行控制向量提取：

```python
# 不同的token位置选项
control_vector = extract_statistical_control_vector(
    method="diffmean",
    all_hidden_states=all_hidden_states,
    positive_indices=positive_indices,
    negative_indices=negative_indices,
    token_pos=-1,      # 最后一个token（默认）
    # token_pos=0,     # 第一个token
    # token_pos="first",  # 第一个token（字符串形式）
    # token_pos="last",   # 最后一个token（字符串形式）
    # token_pos="mean",   # 所有token的均值
    # token_pos="max",    # L2范数最大的token
    # token_pos="min",    # L2范数最小的token
)
```

### Token位置选项说明

| 选项 | 类型 | 说明 |
|------|------|------|
| `-1` | int | 最后一个token（默认，推荐） |
| `0, 1, 2, ...` | int | 指定位置的token |
| `"first"` | str | 第一个token |
| `"last"` | str | 最后一个token |
| `"mean"` | str | 所有token的平均值 |
| `"max"` | str | L2范数最大的token |
| `"min"` | str | L2范数最小的token |

## 输入格式

### all_hidden_states
三维列表结构：`[样本][layer][token]`

```python
all_hidden_states = [
    [  # 样本0
        [token0_layer0, token1_layer0, ...],  # layer 0
        [token0_layer1, token1_layer1, ...],  # layer 1
        ...
    ],
    [  # 样本1
        [token0_layer0, token1_layer0, ...],
        [token0_layer1, token1_layer1, ...],
        ...
    ],
    ...
]
```

每个token的hidden state可以是：
- `torch.Tensor`: 会自动转换为numpy
- `numpy.ndarray`: 直接使用

### 索引列表
```python
positive_indices = [0, 1, 2, 3, 4]  # 正样本在all_hidden_states中的索引
negative_indices = [5, 6, 7, 8, 9]  # 负样本在all_hidden_states中的索引
```

## 输出格式

### StatisticalControlVector
```python
@dataclass
class StatisticalControlVector:
    model_type: str                    # 模型类型
    method: str                        # 提取方法
    directions: dict[int, np.ndarray]  # {layer: direction_vector}
    metadata: dict                     # 元数据
```

### GGUF格式
兼容repeng项目的GGUF格式，可以直接用于llama.cpp等工具。

## 方法特点对比

| 方法 | 需要负样本 | 支持token_pos | 默认token | 特点 |
|------|------------|---------------|-----------|------|
| DiffMean | ✅ | ✅ | 最后一个 | 对比明确，效果好 |
| PCA | ❌ | ✅ | 最后一个 | 主要变化方向 |
| LAT | ❌ | ✅ | 最后一个 | 鲁棒性强 |
| LinearProbe | ✅ | ✅ | 最后一个 | 分类器权重，监督学习 |

## 高级用法

### 自定义参数

```python
# PCA with custom parameters
control_vector = extract_pca_control_vector(
    all_hidden_states=all_hidden_states,
    positive_indices=positive_indices,
    use_positive_only=True,  # 只使用正样本（默认）
    n_components=2,          # PCA组件数
    normalize=True           # 归一化
)

# LAT with custom parameters  
control_vector = extract_lat_control_vector(
    all_hidden_states=all_hidden_states,
    positive_indices=positive_indices,
    use_positive_only=True,  # 只使用正样本
    normalize=True
)
```

## 注意事项

1. **内存使用**: 大模型的隐藏状态会占用大量内存，建议分批处理
2. **数据格式**: 确保all_hidden_states的格式正确
3. **归一化**: 大多数情况下建议开启归一化
4. **层选择**: 不同层的控制向量效果可能差异很大
5. **GGUF兼容**: 输出的GGUF文件与repeng项目完全兼容 