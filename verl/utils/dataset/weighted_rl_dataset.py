# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
import numpy as np
from typing import Optional, Dict, Any
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class WeightedRLHFDataset(RLHFDataset):
    """
    扩展的RLHF数据集，支持动态权重调整和有放回的加权采样。
    
    权重计算公式：w = sqrt(v * (1 - v)) + eps
    其中 v 是reward estimator预测的概率值(0,1)
    """
    
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        eps: float = 0.001,
        initial_v: float = 1,
        min_weight: float = 0.01,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        
        # 初始化权重相关参数
        self.eps = config.get("weight_eps", eps)
        self.initial_v = config.get("initial_v", initial_v)
        self.min_weight = config.get("min_weight", min_weight)
        
        # Epistemic uncertainty configuration
        # Requirements: 3.1, 3.2, 3.3 - Support epistemic uncertainty in sampling weights
        # beta_exploration controls the weight of epistemic uncertainty term
        # When beta_exploration=0, behaves identically to original aleatoric-only formula
        self.beta_exploration = config.get("beta_exploration", 0.5)
        
        # 初始化权重向量
        dataset_size = len(self.dataframe)
        initial_weight = self.initial_v
        self.weights = np.full(dataset_size, initial_weight, dtype=np.float32)
        
        # 存储v值，用于权重计算
        self.v_values = np.full(dataset_size, self.initial_v, dtype=np.float32)
        
        # 存储epistemic uncertainty值 (initialized to 0, will be updated when available)
        self.epistemic_values = np.zeros(dataset_size, dtype=np.float32)
        
        # 为每个样本分配唯一ID
        self.sample_ids = np.arange(dataset_size)
        
        logger.info(f"Initialized WeightedRLHFDataset with {dataset_size} samples, initial weight={initial_weight:.4f}")
        if self.beta_exploration > 0:
            logger.info(f"Epistemic uncertainty enabled with beta_exploration={self.beta_exploration}")
    
    def update_weights_from_v(
        self, 
        indices: np.ndarray, 
        v_values: np.ndarray,
        epistemic_uncertainty: np.ndarray = None
    ):
        """
        根据reward estimator的输出更新权重
        
        When epistemic uncertainty is provided, the weight formula becomes:
        w(x) = sqrt(v̂(1-v̂)) + β·U(x) + ε
        
        This combines:
        - Aleatoric uncertainty: sqrt(v̂(1-v̂)) - higher for predictions near 0.5
        - Epistemic uncertainty: β·U(x) - higher for novel/unfamiliar prompts
        - Smoothing constant: ε - ensures all weights remain positive
        
        Args:
            indices: 样本在数据集中的索引
            v_values: reward estimator预测的概率值 (0,1)
            epistemic_uncertainty: Optional epistemic uncertainty U(x) values.
                                   If provided, will be incorporated into weight computation.
        
        Requirements:
            3.1 - Combine aleatoric and epistemic terms: w(x) = sqrt(v̂(1-v̂)) + β·U(x) + ε
            3.2 - When β=0 or epistemic_uncertainty is None, behave identically to original
            3.3 - High epistemic uncertainty → higher sampling weight
            3.4 - Ensure all weights remain positive by adding smoothing constant ε
            8.3 - Incorporate epistemic uncertainty when enabled
        """
        # 更新v值
        self.v_values[indices] = v_values
        
        # Compute aleatoric uncertainty term: sqrt(v * (1-v))
        # This is maximized when v = 0.5 (most uncertain prediction)
        # Clamp v_values to avoid numerical issues at boundaries
        v_clamped = np.clip(v_values, 1e-7, 1.0 - 1e-7)
        aleatoric_term = np.sqrt(v_clamped * (1 - v_clamped))
        
        # Compute epistemic uncertainty term: β·U(x)
        # Requirements: 3.2 - When epistemic_uncertainty is None, use aleatoric-only formula
        if epistemic_uncertainty is not None and len(epistemic_uncertainty) == len(indices):
            # Get beta_exploration from config (default 0.5)
            beta_exploration = getattr(self, 'beta_exploration', 0.5)
            
            # Requirements: 3.3 - High epistemic uncertainty → higher sampling weight
            epistemic_term = beta_exploration * epistemic_uncertainty
            
            # Store epistemic uncertainty for stats
            if not hasattr(self, 'epistemic_values'):
                self.epistemic_values = np.zeros(len(self.weights), dtype=np.float32)
            self.epistemic_values[indices] = epistemic_uncertainty
            
            logger.debug(f"Incorporating epistemic uncertainty: mean U={epistemic_uncertainty.mean():.4f}, "
                        f"β={beta_exploration}, mean β·U={epistemic_term.mean():.4f}")
        else:
            # Requirements: 3.2 - Backward compatibility when epistemic uncertainty is not available
            epistemic_term = 0.0
        
        # Combine terms: w = sqrt(v*(1-v)) + β·U + ε
        # Requirements: 3.1, 3.4 - Ensure all weights remain positive
        new_weights = aleatoric_term + epistemic_term + self.eps
        new_weights = np.maximum(new_weights, self.min_weight)
        self.weights[indices] = new_weights
        
        logger.debug(f"Updated weights for {len(indices)} samples. Mean weight: {self.weights.mean():.4f}")
    
    def get_sampling_weights(self):
        """
        获取归一化的采样权重（概率分布）
        """
        # 归一化权重作为采样概率
        probabilities = self.weights / self.weights.sum()
        return probabilities
    
    def __getitem__(self, item):
        """重写getitem，添加样本ID信息"""
        row_dict = super().__getitem__(item)
        # 添加样本在数据集中的索引，用于后续权重更新
        row_dict["dataset_index"] = item
        return row_dict
    
    def get_weight_stats(self) -> Dict[str, float]:
        """获取权重统计信息"""
        stats = {
            "weight_mean": float(self.weights.mean()),
            "weight_std": float(self.weights.std()),
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
            "v_mean": float(self.v_values.mean()),
            "v_std": float(self.v_values.std()),
        }
        
        # Include epistemic uncertainty stats if available
        # Requirements: 6.1 - Log epistemic uncertainty metrics
        if hasattr(self, 'epistemic_values') and self.epistemic_values is not None:
            # Only include stats if we have non-zero epistemic values
            if np.any(self.epistemic_values > 0):
                stats["epistemic_mean"] = float(self.epistemic_values.mean())
                stats["epistemic_std"] = float(self.epistemic_values.std())
                stats["epistemic_min"] = float(self.epistemic_values.min())
                stats["epistemic_max"] = float(self.epistemic_values.max())
                stats["beta_exploration"] = float(self.beta_exploration)
        
        return stats
