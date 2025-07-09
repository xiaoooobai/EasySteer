"""
GemmaScope SAE Difference of Means Extractor
基于SAE的均值差方法
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from .utils import StatisticalControlVector

import logging
logger = logging.getLogger(__name__)


class GemmaScopeSAEDiffMeanExtractor:
    """SAE-based difference of means method"""
    
    @staticmethod
    def extract(
        all_hidden_states,
        positive_indices,
        negative_indices=None,
        model_type: str = "unknown",
        sae_encoder=None,  # SAE编码器权重 (hidden_dim, sae_width)
        sae_decoder=None,  # SAE解码器权重 (sae_width, hidden_dim)
        sae_threshold=None,  # SAE阈值 (sae_width,)
        sae_bias_enc=None,  # SAE编码器偏置 (sae_width,)
        normalize: bool = True,
        token_pos: int | str = -1,
        **kwargs
    ) -> StatisticalControlVector:
        """
        Extract control vectors using SAE-based difference of means method.
        
        Args:
            all_hidden_states: 三维列表 [样本][layer][token]
            positive_indices: 正样本的索引列表
            negative_indices: 负样本的索引列表
            model_type: 模型类型名称
            sae_encoder: SAE编码器权重
            sae_decoder: SAE解码器权重
            sae_threshold: SAE阈值
            sae_bias_enc: SAE编码器偏置
            normalize: 是否归一化向量
            token_pos: token位置，-1表示最后一个token（默认），支持int/"first"/"last"/"mean"/"max"/"min"
        """
        if sae_encoder is None or sae_decoder is None:
            raise ValueError("SAE encoder and decoder weights are required for GemmaScopeSAEDiffMean")
        
        # 转换为numpy arrays
        if torch.is_tensor(sae_encoder):
            sae_encoder = sae_encoder.cpu().float().numpy()
        if torch.is_tensor(sae_decoder):
            sae_decoder = sae_decoder.cpu().float().numpy()
        if torch.is_tensor(sae_threshold):
            sae_threshold = sae_threshold.cpu().float().numpy()
        if torch.is_tensor(sae_bias_enc):
            sae_bias_enc = sae_bias_enc.cpu().float().numpy()
        
        if negative_indices is None:
            n_samples = len(all_hidden_states)
            negative_indices = list(range(len(positive_indices), n_samples))
        
        n_layers = len(all_hidden_states[0])
        directions = {}
        
        def sae_encode(activations):
            """SAE编码函数"""
            pre_acts = activations @ sae_encoder
            if sae_bias_enc is not None:
                pre_acts += sae_bias_enc
            
            if sae_threshold is not None:
                mask = pre_acts > sae_threshold
                acts = mask * np.maximum(pre_acts, 0)
            else:
                acts = np.maximum(pre_acts, 0)
            return acts
        
        from .utils import extract_token_hiddens
        
        # 使用新的token提取函数
        positive_hiddens, negative_hiddens = extract_token_hiddens(
            all_hidden_states, positive_indices, negative_indices, token_pos=token_pos
        )
        
        for layer in tqdm(range(n_layers), desc="Computing GemmaScopeSAEDiffMean directions"):
            # 对正样本进行SAE编码
            positive_acts = []
            for hidden in positive_hiddens[layer]:
                acts = sae_encode(hidden.reshape(1, -1))
                positive_acts.append(acts[0])
            positive_acts = np.vstack(positive_acts)
            
            # 对负样本进行SAE编码
            negative_acts = []
            for hidden in negative_hiddens[layer]:
                acts = sae_encode(hidden.reshape(1, -1))
                negative_acts.append(acts[0])
            negative_acts = np.vstack(negative_acts)
            
            # 计算SAE潜在空间的均值
            mean_positive_activation = np.mean(positive_acts, axis=0)
            mean_negative_activation = np.mean(negative_acts, axis=0)
            
            # 计算SAE潜在空间的均值差
            latent_diff = mean_positive_activation - mean_negative_activation
            
            # 解码回原始空间
            direction = latent_diff @ sae_decoder
            
            if normalize:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
            
            directions[layer] = direction.astype(np.float32)
            
            # 记录top特征
            top_indices = np.argsort(np.abs(latent_diff))[-10:][::-1]
            logger.info(f"Layer {layer} top 10 SAE features:")
            for i, idx in enumerate(top_indices):
                logger.info(f"  Feature {idx}: {latent_diff[idx]:.6f}")
        
        metadata = {
            "normalize": normalize,
            "token_pos": token_pos,
            "n_positive": len(positive_indices),
            "n_negative": len(negative_indices),
            "sae_width": sae_decoder.shape[0]
        }
        
        return StatisticalControlVector(
            model_type=model_type,
            method="gemmascope_sae_diffmean",
            directions=directions,
            metadata=metadata
        ) 