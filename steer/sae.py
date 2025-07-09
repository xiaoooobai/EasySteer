#!/usr/bin/env python3
"""
SAE特征提取和分析方法 - 符合steer_method架构
支持 BFloat16 和 Float32 的自动适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Union
import json
from tqdm.auto import tqdm
from .utils import StatisticalControlVector, extract_token_hiddens

import logging
logger = logging.getLogger(__name__)

# 尝试导入safetensors
try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning("safetensors未安装，只支持.npz格式的SAE参数文件")


class SimpleGemmaScopeSAE:
    """简化版的GemmaScope SAE实现 - 支持数据类型自适应"""
    
    def __init__(self):
        self.W_enc = None
        self.W_dec = None
        self.b_enc = None
        self.b_dec = None
        self.threshold = None
        self.device = 'cpu'
        self.dtype = torch.float32  # 默认数据类型
    
    def load_from_params(self, sae_params_path: str, dtype: torch.dtype = None):
        """
        从参数文件加载SAE参数，支持指定数据类型
        
        支持的格式：
        - .npz: numpy存档格式
        - .safetensors: 安全tensor格式（需要安装safetensors）
        """
        # 如果未指定dtype，使用float32
        if dtype is None:
            dtype = torch.float32
        
        self.dtype = dtype
        
        if sae_params_path.endswith('.npz'):
            # 加载.npz格式
            params = np.load(sae_params_path)
            
            # 加载并转换为指定数据类型
            self.W_enc = torch.from_numpy(params['W_enc']).to(dtype)      # [hidden_dim, sae_width]
            self.W_dec = torch.from_numpy(params['W_dec']).to(dtype)      # [sae_width, hidden_dim]
            self.b_enc = torch.from_numpy(params['b_enc']).to(dtype)      # [sae_width]
            self.b_dec = torch.from_numpy(params['b_dec']).to(dtype)      # [hidden_dim]
            self.threshold = torch.from_numpy(params['threshold']).to(dtype)  # [sae_width]
            
            logger.info(f"✓ 加载SAE参数 (npz格式, dtype: {dtype}):")
            
        elif sae_params_path.endswith('.safetensors'):
            # 加载.safetensors格式
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors库未安装，无法加载.safetensors格式文件。请运行: pip install safetensors")
            
            params = safetensors_load(sae_params_path)
            
            # 检查必需的参数
            required_keys = ['W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold']
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                raise ValueError(f"safetensors文件缺少必需的参数: {missing_keys}")
            
            # 加载并转换为指定数据类型
            self.W_enc = params['W_enc'].to(dtype)
            self.W_dec = params['W_dec'].to(dtype)
            self.b_enc = params['b_enc'].to(dtype)
            self.b_dec = params['b_dec'].to(dtype)
            self.threshold = params['threshold'].to(dtype)
            
            logger.info(f"✓ 加载SAE参数 (safetensors格式, dtype: {dtype}):")
            
        else:
            raise ValueError(f"不支持的文件格式: {sae_params_path}。支持的格式: .npz, .safetensors")
        
        logger.info(f"  - 编码维度: {self.W_enc.shape}")
        logger.info(f"  - 解码维度: {self.W_dec.shape}")
        logger.info(f"  - SAE宽度: {self.b_enc.shape[0]}")
        logger.info(f"  - 数据类型: {self.W_enc.dtype}")
    
    def to(self, device, dtype=None):
        """移动到指定设备和数据类型"""
        self.device = device
        if dtype is not None:
            self.dtype = dtype
            
        if self.W_enc is not None:
            self.W_enc = self.W_enc.to(device=device, dtype=self.dtype)
            self.W_dec = self.W_dec.to(device=device, dtype=self.dtype)
            self.b_enc = self.b_enc.to(device=device, dtype=self.dtype)
            self.b_dec = self.b_dec.to(device=device, dtype=self.dtype)
            self.threshold = self.threshold.to(device=device, dtype=self.dtype)
        return self
    
    def _ensure_dtype_compatibility(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """确保hidden_states与SAE参数的数据类型兼容"""
        if hidden_states.dtype != self.dtype:
            logger.warning(f"转换hidden_states数据类型 {hidden_states.dtype} -> {self.dtype}")
            hidden_states = hidden_states.to(self.dtype)
        return hidden_states
    
    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SAE编码：JumpReLU激活"""
        # 确保数据类型兼容
        hidden_states = self._ensure_dtype_compatibility(hidden_states)
        
        # 1. 线性变换 + 偏置
        pre_acts = hidden_states @ self.W_enc + self.b_enc  # [..., sae_width]
        
        # 2. JumpReLU激活 (阈值化ReLU)
        mask = (pre_acts > self.threshold)  # [..., sae_width]
        acts = mask * F.relu(pre_acts)      # [..., sae_width]
        
        return acts
    
    def decode(self, sae_features: torch.Tensor) -> torch.Tensor:
        """SAE解码"""
        reconstructed = sae_features @ self.W_dec + self.b_dec  # [..., hidden_dim]
        return reconstructed
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整的SAE前向传播"""
        sae_features = self.encode(hidden_states)
        reconstructed = self.decode(sae_features)
        return sae_features, reconstructed


class SAEExtractor:
    """SAE特征提取器 - 符合steer_method接口"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        sae_params_path: str = None,
        sae_params: dict = None,
        method: str = "max_diff",
        combination_mode: str = "weighted_all",  # "weighted_all", "weighted_top_k", "single_top_feature"
        top_k: int = 100,  # 当combination_mode="weighted_top_k"时使用的特征数量
        normalize: bool = True,
        token_pos: Union[int, str] = -1,
        target_layer: int = None,
        sae_target_layer: int = None,  # SAE对应的目标层，如果指定则只导出该层
        dtype: torch.dtype = torch.float32,
        **kwargs
    ) -> StatisticalControlVector:
        """
        使用SAE特征分析提取控制向量
        
        Args:
            all_hidden_states: 三维列表 [样本][layer][token]
            positive_indices: 正样本的索引列表
            negative_indices: 负样本的索引列表
            model_type: 模型类型名称
            sae_params_path: SAE参数文件路径(.npz或.safetensors)
            sae_params: SAE参数字典(W_enc, W_dec, b_enc, b_dec, threshold)
            method: 特征分析方法 ("max_diff", "max_activation", "max_auc")
            combination_mode: 特征组合模式
                - "weighted_all": 使用所有特征的加权组合（默认）
                - "weighted_top_k": 使用top_k个特征的加权组合
                - "single_top_feature": 只使用单个最重要特征
            top_k: 当combination_mode="weighted_top_k"时选择的特征数量
            normalize: 是否归一化向量
            token_pos: token位置
            target_layer: 目标层索引，None表示所有层
            sae_target_layer: SAE对应的目标层，如果指定则只导出该层的向量
            dtype: 数据类型
        """
        # 初始化SAE
        sae = SimpleGemmaScopeSAE()
        
        if sae_params_path:
            sae.load_from_params(sae_params_path, dtype=dtype)
        elif sae_params:
            # 从字典加载参数
            sae.dtype = dtype
            sae.W_enc = torch.from_numpy(sae_params['W_enc']).to(dtype)
            sae.W_dec = torch.from_numpy(sae_params['W_dec']).to(dtype)
            sae.b_enc = torch.from_numpy(sae_params['b_enc']).to(dtype)
            sae.b_dec = torch.from_numpy(sae_params['b_dec']).to(dtype)
            sae.threshold = torch.from_numpy(sae_params['threshold']).to(dtype)
        else:
            raise ValueError("必须提供sae_params_path或sae_params参数")
        
        # 如果指定了sae_target_layer，覆盖target_layer
        if sae_target_layer is not None:
            target_layer = sae_target_layer
            logger.info(f"SAE指定目标层: {sae_target_layer}")
        
        # 转换数据格式
        layer_data = convert_to_tensor_format(
            all_hidden_states, positive_indices, negative_indices,
            token_pos=token_pos, target_layer=target_layer
        )
        
        # 分析SAE特征
        sae_results = analyze_sae_features_per_layer(sae, layer_data, method=method)
        
        # 转换为控制向量格式
        directions = {}
        metadata = {
            "analysis_method": method,  # 重命名避免与StatisticalControlVector.method冲突
            "combination_mode": combination_mode,
            "top_k": top_k if combination_mode == "weighted_top_k" else None,
            "normalize": normalize, 
            "token_pos": token_pos,
            "sae_target_layer": sae_target_layer
        }
        
        for layer, result in sae_results.items():
            # 获取特征分数和排序
            feature_scores = result["feature_scores"]  # [sae_width]
            sorted_indices = result["sorted_indices"]  # [sae_width]
            
            # 根据组合模式选择特征
            if combination_mode == "single_top_feature":
                # 模式1：只使用单个最重要特征
                top_feature_idx = sorted_indices[0]
                direction = sae.W_dec[top_feature_idx, :]
                logger.info(f"Layer {layer}: 使用单个特征 {top_feature_idx.item()}, 分数: {feature_scores[top_feature_idx].item():.4f}")
                
            elif combination_mode == "weighted_top_k":
                # 模式2：使用top_k个特征的加权组合
                top_k_indices = sorted_indices[:top_k]
                top_k_scores = feature_scores[top_k_indices]
                
                # 创建稀疏的特征向量
                sparse_scores = torch.zeros_like(feature_scores)
                sparse_scores[top_k_indices] = top_k_scores
                
                if normalize:
                    norm = torch.norm(sparse_scores)
                    if norm > 0:
                        sparse_scores = sparse_scores / norm
                
                direction = sparse_scores @ sae.W_dec
                logger.info(f"Layer {layer}: 使用top {top_k}个特征, 分数范围: {top_k_scores.max().item():.4f} ~ {top_k_scores.min().item():.4f}")
                
            else:  # combination_mode == "weighted_all"
                # 模式3：使用所有特征的加权组合（原始方法）
                if normalize:
                    norm = torch.norm(feature_scores)
                    if norm > 0:
                        feature_scores = feature_scores / norm
                
                direction = feature_scores @ sae.W_dec
                active_features = (torch.abs(feature_scores) > 0.01).sum().item()
                logger.info(f"Layer {layer}: 使用所有特征, 活跃特征数: {active_features}")
            
            directions[layer] = direction.detach().cpu().float().numpy()
            
            # 更新metadata
            metadata.update({
                f"layer_{layer}_active_features": result["active_features"],
                f"layer_{layer}_sparsity": result["sparsity"],
                f"layer_{layer}_top_feature": result["top_10_features"][0],
                f"layer_{layer}_top_score": result["top_10_scores"][0]
            })
        
        return StatisticalControlVector(
            model_type=model_type,
            method=f"sae_{method}_{combination_mode}",
            directions=directions,
            metadata=metadata
        )


def convert_to_tensor_format(all_hidden_states, positive_indices, negative_indices, 
                           token_pos: Union[int, str] = -1, target_layer: int = None):
    """
    将三维列表转换为tensor格式，并提取指定token位置的hidden states
    自动检测和保持数据类型
    """
    def extract_token_position(hidden_states_tokens, pos):
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
            return torch.stack(hidden_states_tokens).mean(dim=0)
        elif pos == "max":
            stacked = torch.stack(hidden_states_tokens)
            return stacked[torch.norm(stacked, dim=-1).argmax()]
        elif pos == "min":
            stacked = torch.stack(hidden_states_tokens)
            return stacked[torch.norm(stacked, dim=-1).argmin()]
        else:
            raise ValueError(f"不支持的token位置: {pos}")
    
    n_layers = len(all_hidden_states[0])
    layers_to_process = [target_layer] if target_layer is not None else range(n_layers)
    
    layer_data = {}
    
    # 检测数据类型（从第一个样本的第一个层的第一个token）
    first_token = all_hidden_states[0][0][0]
    if isinstance(first_token, torch.Tensor):
        detected_dtype = first_token.dtype
        detected_device = first_token.device
        logger.info(f"✓ 检测到hidden states数据类型: {detected_dtype}, 设备: {detected_device}")
    else:
        detected_dtype = torch.float32
        detected_device = torch.device('cpu')
        logger.info(f"✓ Hidden states不是tensor，将转换为: {detected_dtype}")
    
    for layer in layers_to_process:
        # 提取positive样本
        positive_hiddens = []
        for idx in positive_indices:
            if isinstance(all_hidden_states[idx][layer][0], torch.Tensor):
                # 已经是tensor
                token_hidden = extract_token_position(all_hidden_states[idx][layer], token_pos)
            else:
                # 转换为tensor，使用检测到的数据类型
                tokens_tensor = [torch.tensor(t, dtype=detected_dtype) if not isinstance(t, torch.Tensor) 
                               else t.to(detected_dtype) for t in all_hidden_states[idx][layer]]
                token_hidden = extract_token_position(tokens_tensor, token_pos)
            positive_hiddens.append(token_hidden)
        
        # 提取negative样本
        negative_hiddens = []
        for idx in negative_indices:
            if isinstance(all_hidden_states[idx][layer][0], torch.Tensor):
                token_hidden = extract_token_position(all_hidden_states[idx][layer], token_pos)
            else:
                tokens_tensor = [torch.tensor(t, dtype=detected_dtype) if not isinstance(t, torch.Tensor) 
                               else t.to(detected_dtype) for t in all_hidden_states[idx][layer]]
                token_hidden = extract_token_position(tokens_tensor, token_pos)
            negative_hiddens.append(token_hidden)
        
        # 转换为tensor，保持原有的数据类型
        layer_data[layer] = {
            "positive": torch.stack(positive_hiddens),  # [n_pos, hidden_dim]
            "negative": torch.stack(negative_hiddens),  # [n_neg, hidden_dim]
            "dtype": detected_dtype,
            "device": detected_device
        }
    
    return layer_data


def analyze_sae_features_per_layer(
    sae: SimpleGemmaScopeSAE,
    layer_data: Dict,
    method: str = "max_diff"
) -> Dict:
    """
    分析每一层的SAE特征激活 - 支持数据类型自适应
    """
    results = {}
    
    for layer, data in layer_data.items():
        pos_hiddens = data["positive"]  # [n_pos, hidden_dim]
        neg_hiddens = data["negative"]  # [n_neg, hidden_dim]
        
        # 检查数据类型并调整SAE参数
        if hasattr(data, 'dtype') and 'dtype' in data:
            input_dtype = data['dtype']
        else:
            input_dtype = pos_hiddens.dtype
            
        # 如果SAE的数据类型与输入不匹配，调整SAE
        if sae.dtype != input_dtype:
            logger.info(f"调整SAE数据类型: {sae.dtype} -> {input_dtype}")
            sae.to(sae.device, dtype=input_dtype)
        
        # 添加序列维度以适配SAE (假设单token)
        pos_hiddens = pos_hiddens.unsqueeze(1)  # [n_pos, 1, hidden_dim]
        neg_hiddens = neg_hiddens.unsqueeze(1)  # [n_neg, 1, hidden_dim]
        
        # 获取SAE特征激活
        pos_features, _ = sae.forward(pos_hiddens)  # [n_pos, 1, sae_width]
        neg_features, _ = sae.forward(neg_hiddens)  # [n_neg, 1, sae_width]
        
        # 压缩序列维度
        pos_features = pos_features.squeeze(1)  # [n_pos, sae_width]
        neg_features = neg_features.squeeze(1)  # [n_neg, sae_width]
        
        if method == "max_diff":
            # 计算平均激活差异
            pos_mean = pos_features.mean(dim=0)  # [sae_width]
            neg_mean = neg_features.mean(dim=0)  # [sae_width]
            scores = pos_mean - neg_mean
            
        elif method == "max_activation":
            # 使用最大激活值
            pos_max = pos_features.max(dim=0)[0]  # [sae_width]
            neg_max = neg_features.max(dim=0)[0]  # [sae_width]
            scores = pos_max - neg_max
            
        elif method == "max_auc":
            # 计算AUC分数
            try:
                from sklearn.metrics import roc_auc_score
                
                all_features = torch.cat([pos_features, neg_features], dim=0)  # [n_total, sae_width]
                all_labels = torch.cat([
                    torch.ones(pos_features.shape[0]),
                    torch.zeros(neg_features.shape[0])
                ], dim=0)
                
                scores = []
                for i in tqdm(range(all_features.shape[1])):
                    try:
                        # 转换为float32进行sklearn计算
                        features_i = all_features[:, i].detach().cpu().float().numpy()
                        labels = all_labels.detach().cpu().numpy()
                        auc = roc_auc_score(labels, features_i)
                        scores.append(auc)
                    except:
                        scores.append(0.5)
                scores = torch.tensor(scores, dtype=input_dtype)
            except ImportError:
                logger.warning("sklearn不可用，使用max_diff方法")
                pos_mean = pos_features.mean(dim=0)
                neg_mean = neg_features.mean(dim=0)
                scores = pos_mean - neg_mean
        
        # 获取特征排序
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]
        
        # 统计信息
        active_features = (scores > 0.01).sum().item()
        sparsity = 1.0 - (pos_features > 0).float().mean().item()
        
        results[layer] = {
            "feature_scores": scores,
            "sorted_indices": sorted_indices,
            "sorted_scores": sorted_scores,
            "top_10_features": sorted_indices[:10].tolist(),
            "top_10_scores": sorted_scores[:10].float().tolist(),  # 转换为float以便JSON序列化
            "active_features": active_features,
            "sparsity": sparsity,
            "method": method,
            "n_positive": pos_features.shape[0],
            "n_negative": neg_features.shape[0],
            "dtype": str(input_dtype)
        }
    
    return results
