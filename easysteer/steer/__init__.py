"""
Steering Methods Package
统一的控制向量提取方法包
"""

from .utils import StatisticalControlVector, extract_token_hiddens, extract_last_token_hiddens, extract_all_token_hiddens
from .diffmean import DiffMeanExtractor
from .pca import PCAExtractor
from .lat import LATExtractor
from .linear_probe import LinearProbeExtractor
from .unified_interface import (
    extract_statistical_control_vector,
    extract_diffmean_control_vector,
    extract_pca_control_vector,
    extract_lat_control_vector,
    extract_linear_probe_control_vector
)

__all__ = [
    # Core classes
    'StatisticalControlVector',
    
    # Utility functions
    'extract_token_hiddens',
    'extract_last_token_hiddens',
    'extract_all_token_hiddens',
    
    # Extractor classes
    'DiffMeanExtractor',
    'PCAExtractor',
    'LATExtractor',
    'LinearProbeExtractor',
    
    # Unified interface functions
    'extract_statistical_control_vector',
    'extract_diffmean_control_vector',
    'extract_pca_control_vector',
    'extract_lat_control_vector',
    'extract_linear_probe_control_vector'
] 