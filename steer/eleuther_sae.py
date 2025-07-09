import dataclasses
import json
import pathlib
import typing
import os

import numpy as np
import torch
import torch.types
import tqdm
from typing import Dict, Tuple, List, Union
import torch.nn.functional as F
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class SimpleSAE:
    """Simple SAE implementation that loads weights from safetensors files"""
    
    def __init__(self, weights_dict: dict, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # Extract weights from the weights dictionary  
        # Map the actual parameter names to our expected names
        self.W_enc = weights_dict['encoder.weight'].to(device=device, dtype=dtype)  # [d_sae, d_in]
        self.W_dec = weights_dict['W_dec'].to(device=device, dtype=dtype)  # [d_sae, d_in]
        self.b_enc = weights_dict['encoder.bias'].to(device=device, dtype=dtype)  # [d_sae]
        self.b_dec = weights_dict['b_dec'].to(device=device, dtype=dtype)  # [d_in]
        
        self.d_sae = self.W_enc.shape[0]  # 65536
        self.d_in = self.W_enc.shape[1]   # 1536
        
    def encode(self, x):
        """Encode input to SAE features"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
        
        # x shape: (..., d_in)
        # Apply encoding: features = ReLU(W_enc @ (x - b_dec) + b_enc)
        # W_enc: [d_sae, d_in], x: [..., d_in] -> [..., d_sae]
        x_centered = x - self.b_dec
        features = torch.matmul(x_centered, self.W_enc.T) + self.b_enc
        
        # Apply ReLU activation
        features = torch.nn.functional.relu(features)
        
        return features.cpu().numpy()
    
    def decode(self, features):
        """Decode SAE features back to input space"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(device=self.device, dtype=self.dtype)
            
        # Apply decoding: x_recon = W_dec.T @ features + b_dec
        # W_dec: [d_sae, d_in], features: [..., d_sae] -> [..., d_in]
        x_recon = torch.matmul(features, self.W_dec) + self.b_dec
        
        return x_recon.cpu().numpy()
        
    def forward(self, x):
        """Forward pass: encode then decode"""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


@dataclasses.dataclass  
class Sae:
    base_path: str
    layer_pattern: str
    available_layers: List[int]
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_local_path(
        cls,
        base_path: str,
        layers: Union[int, List[int], range],
        device='cpu', 
        dtype=torch.float32,
        layer_pattern: str = "layers.{}.mlp"
    ) -> "Sae":
        """
        从本地路径配置多层SAE，但不立即加载
        
        Args:
            base_path: SAE权重文件的基础路径
            layers: 要检查的层索引
            device: 设备
            dtype: 数据类型
            layer_pattern: 层文件名模式，{}会被层数替换
            
        Returns:
            Sae: 配置好的Sae对象
        """
        if isinstance(layers, int):
            layers_to_check = [layers]
        elif isinstance(layers, range):
            layers_to_check = list(layers)
        else:
            layers_to_check = layers
            
        available_layers = []
        logger.info("Checking for available SAE layers...")
        for layer in tqdm.tqdm(layers_to_check, desc="Checking SAE layers"):
            layer_name = layer_pattern.format(layer)
            layer_dir = os.path.join(base_path, layer_name)
            weights_path = os.path.join(layer_dir, "sae.safetensors")
            if os.path.exists(weights_path):
                available_layers.append(layer)
            else:
                logger.warning(f"Weights file not found for layer {layer}: {weights_path}, skipping.")

        if not available_layers:
            raise ValueError("No SAE layers found at the specified path with the given pattern.")
            
        logger.info(f"Found {len(available_layers)} available SAE layers: {available_layers}")
        
        return cls(
            base_path=base_path,
            layer_pattern=layer_pattern,
            available_layers=available_layers,
            device=device,
            dtype=dtype
        )

    def _load_single_sae(self, layer_idx: int) -> SimpleSAE:
        """为单层加载SAE"""
        layer_name = self.layer_pattern.format(layer_idx)
        layer_dir = os.path.join(self.base_path, layer_name)
        weights_path = os.path.join(layer_dir, "sae.safetensors")
        config_path = os.path.join(layer_dir, "cfg.json")
        
        logger.debug(f"Loading SAE for layer {layer_idx} from {weights_path}")
        sae = load_sae_from_safetensors(weights_path, config_path, device=self.device, dtype=self.dtype)
        return sae
    
    @staticmethod
    def _extract_token_position(hidden_states_tokens, pos):
        """提取指定位置的token hidden states"""
        if isinstance(pos, int):
            if pos == -1:
                return hidden_states_tokens[-1]  # 最后一个token
            else:
                return hidden_states_tokens[pos]  # 指定位置
        elif pos == "first":
            return hidden_states_tokens[0]
        elif pos == "last":
            return hidden_states_tokens[-1]
        elif pos == "mean":
            if isinstance(hidden_states_tokens[0], np.ndarray):
                return np.stack(hidden_states_tokens).mean(axis=0)
            else:  # torch.Tensor
                return torch.stack(hidden_states_tokens).mean(dim=0)
        elif pos == "max":
            if isinstance(hidden_states_tokens[0], np.ndarray):
                stacked = np.stack(hidden_states_tokens)
                return stacked[np.linalg.norm(stacked, axis=-1).argmax()]
            else:  # torch.Tensor
                stacked = torch.stack(hidden_states_tokens)
                return stacked[torch.norm(stacked, dim=-1).argmax()]
        elif pos == "min":
            if isinstance(hidden_states_tokens[0], np.ndarray):
                stacked = np.stack(hidden_states_tokens)
                return stacked[np.linalg.norm(stacked, axis=-1).argmin()]
            else:  # torch.Tensor
                stacked = torch.stack(hidden_states_tokens)
                return stacked[torch.norm(stacked, dim=-1).argmin()]
        else:
            raise ValueError(f"不支持的token位置: {pos}")

    def _analyze_layer_features(self, sae, data, method="max_diff"):
        """分析单层的SAE特征激活"""
        # 编码positive和negative样本  
        pos_features = sae.encode(data['positive'])
        neg_features = sae.encode(data['negative'])
        
        weights = None

        # 计算特征重要性权重
        if method == "max_diff":
            # 计算平均激活差异
            pos_mean = np.mean(pos_features, axis=0)
            neg_mean = np.mean(neg_features, axis=0)
            weights = pos_mean - neg_mean
            
        elif method == "max_activation":
            # 使用最大激活值
            pos_mean = np.mean(pos_features, axis=0)
            neg_mean = np.mean(neg_features, axis=0)
            weights = np.maximum(pos_mean, neg_mean)
            
        elif method == "max_auc":
            # 计算AUC分数(需要sklearn)
            try:
                from sklearn.metrics import roc_auc_score
                auc_weights = []
                for feat_idx in range(pos_features.shape[1]):
                    pos_vals = pos_features[:, feat_idx]
                    neg_vals = neg_features[:, feat_idx]
                    
                    all_vals = np.concatenate([pos_vals, neg_vals])
                    labels = np.concatenate([np.ones(len(pos_vals)), np.zeros(len(neg_vals))])
                    
                    try:
                        auc = roc_auc_score(labels, all_vals)
                        auc_weights.append(abs(auc - 0.5))  # 距离随机分类的距离
                    except:
                        auc_weights.append(0.0)
                weights = np.array(auc_weights)
            except ImportError:
                logger.warning("sklearn not available, falling back to max_diff method")
                pos_mean = np.mean(pos_features, axis=0)  
                neg_mean = np.mean(neg_features, axis=0)
                weights = np.abs(pos_mean - neg_mean)
        
        elif method == "pca_diff":
            # 在SAE特征差异上执行PCA
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                logger.error("scikit-learn is not installed. `pip install scikit-learn` to use `pca_diff` method.")
                raise

            min_samples = min(len(pos_features), len(neg_features))
            differences = []
            
            if min_samples > 0:
                differences.append(pos_features[:min_samples] - neg_features[:min_samples])
            
            if len(pos_features) > min_samples:
                neg_mean = np.mean(neg_features, axis=0) if len(neg_features) > 0 else np.zeros_like(pos_features[0])
                differences.append(pos_features[min_samples:] - neg_mean)
            
            if len(neg_features) > min_samples:
                pos_mean = np.mean(pos_features, axis=0) if len(pos_features) > 0 else np.zeros_like(neg_features[0])
                differences.append(pos_mean - neg_features[min_samples:])

            if not differences:
                logger.warning(f"No differences to compute PCA on, using zero vector.")
                weights = np.zeros(sae.d_sae)
            else:
                all_diffs = np.vstack(differences)
                pca = PCA(n_components=1)
                pca.fit(all_diffs)
                weights = pca.components_[0]
            
                # 方向校正
                if len(pos_features) > 0 and len(neg_features) > 0:
                    # 使用成对比较来校正方向，更鲁棒
                    proj_pos = pos_features @ weights
                    proj_neg = neg_features @ weights
                    
                    min_len = min(len(proj_pos), len(proj_neg))
                    positive_larger_mean = np.mean(proj_pos[:min_len] > proj_neg[:min_len])

                    if positive_larger_mean < 0.5:
                        weights *= -1
                        logger.info(f"SAE PCA direction corrected (flipped)")
                            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return weights, pos_features, neg_features

    def extract_features(
        self,
        all_hidden_states,
        positive_indices,
        negative_indices,
        model_type: str = "unknown",
        method: str = "max_diff",
        combination_mode: str = "weighted_all", 
        top_k: int = 10,
        normalize: bool = True,
        token_pos: int = -1,
        target_layer: int = None,
        **kwargs
    ):
        """
        使用多层SAE特征分析提取控制向量, 优化内存使用
        
        Args:
            all_hidden_states: 所有层的hidden states
            positive_indices: 正样本索引
            negative_indices: 负样本索引
            model_type: 模型类型
            method: 分析方法 ('max_diff', 'max_activation', 'max_auc', 'pca_diff')  
            combination_mode: 组合模式 ('weighted_all', 'weighted_top_k', 'single_top_feature')
            top_k: 保留的顶部特征数量
            normalize: 是否标准化
            token_pos: token位置
            target_layer: 目标层索引，None表示所有可用层
        """
        n_hidden_layers = len(all_hidden_states[0])

        if target_layer is not None:
            if target_layer in self.available_layers and target_layer < n_hidden_layers:
                layers_to_process = [target_layer]
            else:
                raise ValueError(
                    f"指定的target_layer {target_layer} 不可用. "
                    f"可用的SAE层: {self.available_layers}, 模型中的可用层: {list(range(n_hidden_layers))}"
                )
        else:
            layers_to_process = [layer for layer in self.available_layers if layer < n_hidden_layers]
            
        logger.info(f"将处理 {len(layers_to_process)} 层: {layers_to_process}")
        
        directions = {}
        
        for layer_idx in tqdm.tqdm(layers_to_process, desc="Extracting vectors per layer"):
            # 1. 动态加载SAE
            sae = self._load_single_sae(layer_idx)
            
            # 2. 提取该层的hidden states
            positive_hiddens = []
            for idx in positive_indices:
                token_hidden = self._extract_token_position(all_hidden_states[idx][layer_idx], token_pos)
                if isinstance(token_hidden, torch.Tensor):
                    token_hidden = token_hidden.cpu().float().numpy()
                positive_hiddens.append(token_hidden)
                
            negative_hiddens = []
            for idx in negative_indices:
                token_hidden = self._extract_token_position(all_hidden_states[idx][layer_idx], token_pos)
                if isinstance(token_hidden, torch.Tensor):
                    token_hidden = token_hidden.cpu().float().numpy()
                negative_hiddens.append(token_hidden)
                
            layer_data = {
                "positive": np.stack(positive_hiddens) if positive_hiddens else np.array([]),
                "negative": np.stack(negative_hiddens) if negative_hiddens else np.array([]),
            }

            # 3. 分析SAE特征
            sae_weights, _, _ = self._analyze_layer_features(sae, layer_data, method=method)
            
            # 4. 根据combination_mode组合特征
            feature_vector = np.zeros(sae.d_sae)
            if combination_mode == "weighted_all":
                feature_vector = sae_weights
            elif combination_mode == "weighted_top_k":
                top_indices = np.argsort(np.abs(sae_weights))[-top_k:]
                feature_vector[top_indices] = sae_weights[top_indices]
            elif combination_mode == "single_top_feature":
                top_idx = np.argmax(np.abs(sae_weights))
                feature_vector[top_idx] = sae_weights[top_idx]
            else:
                raise ValueError(f"Unknown combination_mode: {combination_mode}")

            # 5. 将特征空间的向量解码回原始空间
            if np.any(feature_vector): # 只有在非零向量时才解码
                control_direction = sae.decode(feature_vector.reshape(1, -1))[0]
            else:
                control_direction = np.zeros(sae.d_in)

            # 6. 标准化
            if normalize:
                norm = np.linalg.norm(control_direction)
                if norm > 0:
                    control_direction = control_direction / norm
                    
            directions[layer_idx] = control_direction

            # 7. 释放SAE模型和清理缓存
            del sae
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                
        # 创建StatisticalControlVector对象
        return StatisticalControlVector(
            model_type=model_type,
            method=f"sae_{method}_{combination_mode}",
            directions=directions
        )


def load_sae_from_safetensors(weights_path: str, config_path: str, device='cpu', dtype=torch.float32):
    """从safetensors文件加载单个SAE"""
    
    try:
        import safetensors.torch
    except ImportError as e:
        raise ImportError(
            "safetensors not installed. Please install with `pip install safetensors`"
        ) from e
    
    # 加载权重
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    weights_dict = safetensors.torch.load_file(weights_path)
    
    # 加载配置
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using default config")
        config = {}
    
    return SimpleSAE(weights_dict, device=device, dtype=dtype)


def from_local_path(
    base_path: str,
    layers: Union[int, List[int], range],
    device='cpu', 
    dtype=torch.float32,
    layer_pattern: str = "layers.{}.mlp"
) -> Sae:
    """
    从本地路径加载多层SAE权重
    
    Args:
        base_path: SAE权重文件的基础路径
        layers: 要加载的层索引
        device: 设备
        dtype: 数据类型
        layer_pattern: 层文件名模式，{}会被层数替换
        
    Returns:
        Sae: 包含多层SAE的对象
    """
    return Sae.from_local_path(
        base_path=base_path,
        layers=layers,
        device=device,
        dtype=dtype,
        layer_pattern=layer_pattern
    )