# SPDX-License-Identifier: Apache-2.0
"""
Base adapter interface for LLM frameworks
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch.nn as nn


class LLMAdapter(ABC):
    """
    Abstract base class for LLM framework adapters
    
    This interface provides a unified way to interact with different LLM frameworks
    (vLLM, HuggingFace Transformers, etc.) for hidden states capture.
    """
    
    @abstractmethod
    def extract_model(self, llm: Any) -> nn.Module:
        """
        Extract the underlying PyTorch model from the LLM instance
        
        Args:
            llm: The LLM instance
            
        Returns:
            The PyTorch model (nn.Module)
        """
        pass
    
    @abstractmethod
    def get_model_name(self, llm: Any) -> str:
        """
        Get the model name/path from the LLM instance
        
        Args:
            llm: The LLM instance
            
        Returns:
            Model name or path string
        """
        pass
    
    @abstractmethod
    def encode_texts(self, llm: Any, texts: List[str]) -> Any:
        """
        Encode texts using the LLM
        
        Args:
            llm: The LLM instance
            texts: List of input texts
            
        Returns:
            LLM outputs (format depends on the framework)
        """
        pass
    
    def get_pooling_metadata(self, llm: Any) -> Optional[Any]:
        """
        Get pooling metadata for sample mapping (optional)
        
        Args:
            llm: The LLM instance
            
        Returns:
            Pooling metadata if available, None otherwise
        """
        return None
    
    def setup_capture_context(self, llm: Any) -> Optional[Any]:
        """
        Setup any framework-specific context for capture (optional)
        
        Args:
            llm: The LLM instance
            
        Returns:
            Context object if needed, None otherwise
        """
        return None
    
    def cleanup_capture_context(self, llm: Any, context: Any) -> None:
        """
        Clean up framework-specific context after capture (optional)
        
        Args:
            llm: The LLM instance
            context: Context object from setup_capture_context
        """
        pass
    
    def supports_multi_batch(self) -> bool:
        """
        Whether this adapter supports multi-batch capture
        
        Returns:
            True if multi-batch capture is supported
        """
        return False
    
    def estimate_sample_lengths(self, llm: Any, outputs: Any) -> Optional[List[int]]:
        """
        Estimate the token lengths of each sample from outputs (optional)
        
        Args:
            llm: The LLM instance
            outputs: LLM outputs
            
        Returns:
            List of token lengths for each sample, None if cannot estimate
        """
        return None 