"""
Unified Interface for Steering Methods
统一的控制向量提取方法接口
"""

from .utils import StatisticalControlVector
from .diffmean import DiffMeanExtractor
from .pca import PCAExtractor
from .lat import LATExtractor
from .gemmascope_sae_diffmean import GemmaScopeSAEDiffMeanExtractor
from .linear_probe import LinearProbeExtractor
from .sae import SAEExtractor


def extract_statistical_control_vector(
    method: str,
    all_hidden_states,
    positive_indices,
    negative_indices=None,
    **kwargs
) -> StatisticalControlVector:
    """
    统一的控制向量提取接口
    
    Args:
        method: 方法名称，支持的方法：
               "diffmean", "pca", "lat", "gemmascope_sae_diffmean", "linear_probe", "sae"
        all_hidden_states: 三维列表 [样本][layer][token]
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表
        **kwargs: 方法特定的参数
    
    Returns:
        StatisticalControlVector: 提取的控制向量
    """
    method_map = {
        "diffmean": DiffMeanExtractor,
        "pca": PCAExtractor,
        "lat": LATExtractor,
        "gemmascope_sae_diffmean": GemmaScopeSAEDiffMeanExtractor,
        "linear_probe": LinearProbeExtractor,
        "sae": SAEExtractor,
    }
    
    if method not in method_map:
        supported_methods = list(method_map.keys())
        raise ValueError(f"不支持的方法: {method}。支持的方法: {supported_methods}")
    
    extractor_class = method_map[method]
    return extractor_class.extract(
        all_hidden_states=all_hidden_states,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        **kwargs
    )


def extract_diffmean_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """提取DiffMean控制向量"""
    return DiffMeanExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_pca_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    提取PCA控制向量
    
    Args:
        all_hidden_states: 三维列表 [样本][layer][token]
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表
        use_positive_only: 是否只使用正样本，默认True
            - True: 只使用正样本（传统PCA）
            - False: 先计算正负样本差值，再进行PCA
        correct_direction: 是否校正向量方向（确保从负样本指向正样本），默认True
        n_components: PCA组件数量，默认1
        normalize: 是否归一化向量，默认True
        token_pos: token位置，默认-1（最后一个token）
        **kwargs: 其他参数
    
    Returns:
        StatisticalControlVector: PCA控制向量
    
    Examples:
        >>> # 只使用正样本（传统PCA）
        >>> pca_vector = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices,
        ...     use_positive_only=True
        ... )
        >>> 
        >>> # 使用正负样本差值进行PCA，启用方向校正（默认）
        >>> pca_diff_vector = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=True
        ... )
        >>> 
        >>> # 使用正负样本差值进行PCA，关闭方向校正
        >>> pca_diff_no_correct = extract_pca_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=False
        ... )
    """
    return PCAExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_lat_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    提取LAT控制向量
    
    Args:
        all_hidden_states: 三维列表 [样本][layer][token]
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表
        use_positive_only: 是否只使用正样本，默认True
        correct_direction: 是否校正向量方向（确保从负样本指向正样本），默认True
        n_components: PCA组件数量，默认1
        normalize: 是否归一化向量，默认True
        token_pos: token位置，默认-1（最后一个token）
        **kwargs: 其他参数
    
    Returns:
        StatisticalControlVector: LAT控制向量
    
    Examples:
        >>> # 只使用正样本（传统LAT）
        >>> lat_vector = extract_lat_control_vector(
        ...     all_hidden_states, positive_indices,
        ...     use_positive_only=True
        ... )
        >>> 
        >>> # 使用正负样本，启用方向校正（默认）
        >>> lat_mixed_vector = extract_lat_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     use_positive_only=False, correct_direction=True
        ... )
    """
    return LATExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_gemmascope_sae_diffmean_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """提取GemmaScope SAE DiffMean控制向量"""
    return GemmaScopeSAEDiffMeanExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_linear_probe_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    提取Linear Probe控制向量
    
    Args:
        all_hidden_states: 三维列表 [样本][layer][token]
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表
        regularization: 正则化类型 ("l1", "l2", "elasticnet", "none")，默认"l2"
        C: 正则化强度的倒数，默认1.0。注意：
           - L2正则化: C=1.0通常合适
           - L1正则化: 建议C=10.0或更大，避免过度稀疏化
           - 无正则化: C参数被忽略
        standardize: 是否标准化特征，默认True
        **kwargs: 其他参数
    
    Returns:
        StatisticalControlVector: Linear Probe控制向量
    
    Example:
        >>> # L2正则化（推荐）
        >>> linear_probe_vector = extract_linear_probe_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", regularization="l2", C=1.0
        ... )
        >>> 
        >>> # L1正则化（特征选择）
        >>> linear_probe_l1 = extract_linear_probe_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", regularization="l1", C=10.0
        ... )
    """
    return LinearProbeExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_sae_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """
    提取SAE特征分析控制向量
    
    Args:
        all_hidden_states: 三维列表 [样本][layer][token]
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表
        sae_params_path: SAE参数文件路径(.npz)
        sae_params: SAE参数字典(W_enc, W_dec, b_enc, b_dec, threshold)
        method: 特征分析方法 ("max_diff", "max_activation", "max_auc")，默认"max_diff"
        combination_mode: 特征组合模式，默认"weighted_all"
            - "weighted_all": 使用所有特征的加权组合
            - "weighted_top_k": 使用top_k个特征的加权组合  
            - "single_top_feature": 只使用单个最重要特征
        top_k: 当combination_mode="weighted_top_k"时选择的特征数量，默认100
        normalize: 是否归一化向量，默认True
        token_pos: token位置，默认-1（最后一个token）
        target_layer: 目标层索引，None表示所有层
        sae_target_layer: SAE对应的目标层，如果指定则只导出该层的向量
        dtype: 数据类型，默认torch.float32
        **kwargs: 其他参数
    
    Returns:
        StatisticalControlVector: SAE特征分析控制向量
    
    Examples:
        >>> # 使用单个最重要特征（类似原始axbench实现）
        >>> sae_vector = extract_sae_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", sae_params_path="sae_params.npz",
        ...     combination_mode="single_top_feature", sae_target_layer=15
        ... )
        >>> 
        >>> # 使用top 50个特征的加权组合
        >>> sae_vector = extract_sae_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", sae_params_path="sae_params.npz",
        ...     combination_mode="weighted_top_k", top_k=50
        ... )
        >>> 
        >>> # 使用所有特征的加权组合（默认）
        >>> sae_vector = extract_sae_control_vector(
        ...     all_hidden_states, positive_indices, negative_indices,
        ...     model_type="qwen2.5", sae_params_path="sae_params.npz"
        ... )
    """
    return SAEExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs) 