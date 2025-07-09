# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Storage

Core storage functionality for capturing and managing hidden states from transformer layers.
"""

from typing import Dict, List, Optional, Any
import torch
import threading


class HiddenStatesStore:
    """Global storage for hidden states from transformer layers"""

    def __init__(self):
        self.hidden_states: Dict[int, torch.Tensor] = {}  # layer_id -> hidden_state
        self.layer_names: Dict[int, str] = {}  # layer_id -> layer_name
        self.capture_enabled = False
        self.lock = threading.Lock()
        self.pooling_metadata = None  # Store pooling metadata for sample mapping
        
        # Multi-batch support
        self.batch_hidden_states: List[Dict[int, torch.Tensor]] = []  # List of batches
        self.batch_pooling_metadata: List[Any] = []  # Metadata for each batch
        self.multi_batch_mode = False  # Whether we're in multi-batch capture mode
        self.finalized = False  # Whether multi-batch finalization is complete

    def clear(self):
        """Clear all stored hidden states"""
        with self.lock:
            self.hidden_states.clear()
            self.layer_names.clear()
            self.pooling_metadata = None
            self.batch_hidden_states.clear()
            self.batch_pooling_metadata.clear()
            self.multi_batch_mode = False
            self.finalized = False
            
            # 显式清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def enable_capture(self):
        """Enable hidden states capture"""
        with self.lock:
            self.capture_enabled = True

    def disable_capture(self):
        """Disable hidden states capture"""
        with self.lock:
            self.capture_enabled = False

    def store_hidden_state(self, layer_id: int, hidden_state: torch.Tensor, layer_name: str = ""):
        """Store hidden state for a specific layer"""
        if not (self.capture_enabled and isinstance(hidden_state, torch.Tensor)):
            return
            
        with self.lock:
            # 如果已经finalized，不要覆盖合并后的数据
            if self.finalized:
                return
            
            # 在多批次模式下，检测新的forward调用
            # 当第一层再次出现时，说明开始了新的forward调用（新批次）
            if (self.multi_batch_mode and 
                layer_id == 0 and 
                layer_id in self.hidden_states):
                
                # 结束当前批次，准备新批次
                self.finish_current_batch()
            
            # 将hidden state移动到CPU以节省GPU内存，并创建副本以避免后续计算修改原始数据
            cpu_hidden_state = hidden_state.detach().cpu().clone()
            self.hidden_states[layer_id] = cpu_hidden_state
            self.layer_names[layer_id] = layer_name

    def get_all_hidden_states(self, device: str = 'cpu') -> List[torch.Tensor]:
        """
        Get all hidden states in layer order
        
        Args:
            device: Target device for the returned tensors ('cpu' or 'cuda')
                   Default is 'cpu' to save GPU memory
        """
        with self.lock:
            sorted_layers = sorted(self.hidden_states.keys())
            result = []
            
            target_device = torch.device(device)
            for layer_id in sorted_layers:
                tensor = self.hidden_states[layer_id]
                if device != 'cpu' and tensor.device != target_device:
                    tensor = tensor.to(target_device)
                result.append(tensor)
                
            return result

    def get_hidden_state(self, layer_id: int) -> Optional[torch.Tensor]:
        """Get hidden state for a specific layer"""
        with self.lock:
            return self.hidden_states.get(layer_id)

    def get_layer_count(self) -> int:
        """Get the number of captured layers"""
        with self.lock:
            return len(self.hidden_states)

    def get_layer_info(self) -> Dict[int, str]:
        """Get layer ID to name mapping"""
        with self.lock:
            return self.layer_names.copy()
    
    def get_pooling_metadata(self):
        """Get captured pooling metadata"""
        with self.lock:
            return self.pooling_metadata
    
    def set_pooling_metadata(self, metadata):
        """Set pooling metadata"""
        with self.lock:
            if not self.finalized:  # Only set if not finalized
                self.pooling_metadata = metadata
    
    def enable_multi_batch_mode(self):
        """Enable multi-batch capture mode for handling large batches split by frameworks"""
        with self.lock:
            self.multi_batch_mode = True
    
    def finish_current_batch(self):
        """Mark the current batch as finished and prepare for the next batch"""
        # Note: 此方法假设已经在lock内部被调用
        if self.multi_batch_mode and self.hidden_states:
            # Store current batch
            self.batch_hidden_states.append(self.hidden_states.copy())
            self.batch_pooling_metadata.append(self.pooling_metadata)
            
            # Clear for next batch
            self.hidden_states.clear()
            self.pooling_metadata = None
    
    def finalize_multi_batch(self):
        """Finalize multi-batch capture by combining all batches"""
        with self.lock:
            if not self.multi_batch_mode:
                return
                
            # Add any remaining hidden states as the final batch
            if self.hidden_states:
                self.finish_current_batch()
            
            # 如果没有收集到任何批次，说明是单批次情况
            if not self.batch_hidden_states:
                return
            
            # 如果只有一个批次，直接使用它
            if len(self.batch_hidden_states) == 1:
                self.hidden_states = self.batch_hidden_states[0]
                self.pooling_metadata = self.batch_pooling_metadata[0]
            else:
                # 多个批次，需要合并
                combined_hidden_states = {}
                
                # Get all layer IDs from all batches
                all_layer_ids = set()
                for batch in self.batch_hidden_states:
                    all_layer_ids.update(batch.keys())
                
                # Combine each layer across batches
                # 使用GPU进行合并操作以提高效率
                device = None
                for layer_id in sorted(all_layer_ids):
                    layer_tensors = []
                    for batch in self.batch_hidden_states:
                        if layer_id in batch:
                            tensor = batch[layer_id]
                            # 确定目标设备（使用第一个tensor的设备）
                            if device is None:
                                # 检查是否有可用的GPU
                                if torch.cuda.is_available():
                                    device = torch.device('cuda')
                                else:
                                    device = tensor.device
                            # 将tensor移动到目标设备进行合并
                            layer_tensors.append(tensor.to(device))
                    
                    if layer_tensors:
                        # 在GPU上进行拼接操作，然后立即移回CPU
                        combined_tensor = torch.cat(layer_tensors, dim=0)
                        combined_hidden_states[layer_id] = combined_tensor.cpu()
                        
                        # 清理GPU内存
                        del layer_tensors
                        if device and device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                self.hidden_states = combined_hidden_states
                
                # Combine pooling metadata - use the first non-None metadata
                if self.batch_pooling_metadata:
                    for metadata in self.batch_pooling_metadata:
                        if metadata is not None:
                            self.pooling_metadata = metadata
                            break
            
            # Clean up batch data and force garbage collection
            self.batch_hidden_states.clear()
            self.batch_pooling_metadata.clear()
            self.multi_batch_mode = False
            
            # 显式清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Mark as finalized to prevent further overwrites
            self.finalized = True


class HiddenStatesCaptureContext:
    """Context manager for enabling/disabling hidden states capture"""

    def __init__(self, store: Optional[HiddenStatesStore] = None, multi_batch_mode: bool = False):
        self.store = store if store is not None else _get_global_store()
        self.was_enabled = False
        self.multi_batch_mode = multi_batch_mode

    def __enter__(self):
        self.was_enabled = self.store.capture_enabled
        self.store.enable_capture()
        self.store.clear()
        
        if self.multi_batch_mode:
            self.store.enable_multi_batch_mode()
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.multi_batch_mode and not self.store.finalized:
            self.store.finalize_multi_batch()
            
        if not self.was_enabled:
            self.store.disable_capture()

    def finish_batch(self):
        """Mark current batch as finished (for multi-batch mode)"""
        if self.multi_batch_mode:
            self.store.finish_current_batch()

    def get_all_hidden_states(self, device: str = 'cpu') -> List[torch.Tensor]:
        """获取所有捕获的hidden states"""
        return self.store.get_all_hidden_states(device)

    def get_hidden_state(self, layer_id: int) -> Optional[torch.Tensor]:
        """获取指定层的hidden states"""
        return self.store.get_hidden_state(layer_id)

    def get_layer_count(self) -> int:
        """获取捕获的层数"""
        return self.store.get_layer_count()

    def get_layer_info(self) -> Dict[int, str]:
        """获取层信息"""
        return self.store.get_layer_info()


# Global store instance
_global_hidden_states_store = None

def _get_global_store() -> HiddenStatesStore:
    """Get or create the global hidden states store"""
    global _global_hidden_states_store
    if _global_hidden_states_store is None:
        _global_hidden_states_store = HiddenStatesStore()
    return _global_hidden_states_store

def get_global_store() -> HiddenStatesStore:
    """Public function to get the global store"""
    return _get_global_store() 