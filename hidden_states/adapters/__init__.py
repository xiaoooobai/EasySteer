# SPDX-License-Identifier: Apache-2.0
"""
Adapters for LLM frameworks
"""

from .base import LLMAdapter
from .vllm import VLLMAdapter

__all__ = [
    "LLMAdapter",
    "VLLMAdapter", 
] 