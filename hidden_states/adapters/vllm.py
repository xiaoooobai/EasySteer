# SPDX-License-Identifier: Apache-2.0
"""
vLLM adapter for hidden states capture
"""

from typing import Any, List, Optional
import torch.nn as nn
from .base import LLMAdapter


class VLLMAdapter(LLMAdapter):
    """
    Adapter for vLLM framework
    """
    
    def extract_model(self, llm: Any) -> nn.Module:
        """Extract PyTorch model from vLLM LLM instance"""
        try:
            # Try vLLM v1 structure first
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
                model_executor = llm.llm_engine.model_executor
                if hasattr(model_executor, 'driver_worker'):
                    model = model_executor.driver_worker.model_runner.model
                else:
                    model = getattr(model_executor, 'model', None)
                
                if model is not None:
                    return model
            
            # Try direct model access
            if hasattr(llm, 'model'):
                return llm.model
                
            raise ValueError("Could not extract model from vLLM instance")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract model from vLLM instance: {str(e)}")
    
    def get_model_name(self, llm: Any) -> str:
        """Get model name from vLLM LLM instance"""
        try:
            # Try to get from model config
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_config'):
                return getattr(llm.llm_engine.model_config, 'model', '')
            
            # Fallback to empty string
            return ""
            
        except Exception:
            return ""
    
    def encode_texts(self, llm: Any, texts: List[str]) -> Any:
        """Encode texts using vLLM"""
        return llm.encode(texts)
    
    def get_pooling_metadata(self, llm: Any) -> Optional[Any]:
        """Get pooling metadata from vLLM forward context"""
        try:
            # Try to import vLLM's forward context
            from vllm.forward_context import get_forward_context
            
            forward_context = get_forward_context()
            if hasattr(forward_context, 'attn_metadata') and forward_context.attn_metadata:
                attn_metadata = forward_context.attn_metadata
                if hasattr(attn_metadata, 'pooling_metadata'):
                    return attn_metadata.pooling_metadata
        except ImportError:
            # vLLM not available or different version
            pass
        except Exception:
            # Other errors - ignore silently
            pass
        
        return None
    
    def supports_multi_batch(self) -> bool:
        """vLLM supports multi-batch capture for large inputs"""
        return True
    
    def estimate_sample_lengths(self, llm: Any, outputs: Any) -> Optional[List[int]]:
        """Estimate sample lengths from vLLM outputs"""
        if not outputs:
            return None
        
        sample_lengths = []
        for output in outputs:
            if hasattr(output, 'prompt_token_ids'):
                # EmbeddingRequestOutput format
                sample_lengths.append(len(output.prompt_token_ids))
            elif hasattr(output, 'token_ids'):
                # Other output formats
                sample_lengths.append(len(output.token_ids))
            else:
                # Fallback: estimate from string representation
                estimated_length = max(1, len(str(output).split()) + 5)
                sample_lengths.append(estimated_length)
        
        return sample_lengths if sample_lengths else None 