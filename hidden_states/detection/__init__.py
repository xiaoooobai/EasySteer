# SPDX-License-Identifier: Apache-2.0
"""
Model detection and pattern matching
"""

from .patterns import get_layer_patterns_for_model
from .structure import detect_layer_structure, get_optimal_layer_pattern

__all__ = [
    "get_layer_patterns_for_model",
    "detect_layer_structure", 
    "get_optimal_layer_pattern",
] 