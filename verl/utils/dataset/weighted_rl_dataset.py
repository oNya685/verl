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
        initial_v: float = 0.5,
        min_weight: float = 0.01,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        
        # 初始化权重相关参数
        self.eps = config.get("weight_eps", eps)
        self.initial_v = config.get("initial_v", initial_v)
        self.min_weight = config.get("min_weight", min_weight)
        
        # 初始化权重向量
        dataset_size = len(self.dataframe)
        initial_weight = np.sqrt(self.initial_v * (1 - self.initial_v)) + self.eps
        self.weights = np.full(dataset_size, initial_weight, dtype=np.float32)
        
        # 存储v值，用于权重计算
        self.v_values = np.full(dataset_size, self.initial_v, dtype=np.float32)
        
        # 为每个样本分配唯一ID
        self.sample_ids = np.arange(dataset_size)
        
        logger.info(f"Initialized WeightedRLHFDataset with {dataset_size} samples, initial weight={initial_weight:.4f}")
    
    def update_weights_from_v(self, indices: np.ndarray, v_values: np.ndarray):
        """
        根据reward estimator的输出更新权重
        
        Args:
            indices: 样本在数据集中的索引
            v_values: reward estimator预测的概率值 (0,1)
        """
        # 更新v值
        self.v_values[indices] = v_values
        
        # 计算新权重: w = sqrt(v * (1-v)) + eps
        # 确保权重不低于最小值，避免某些样本永远不被采样
        new_weights = np.sqrt(v_values * (1 - v_values)) + self.eps
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
        return {
            "weight_mean": float(self.weights.mean()),
            "weight_std": float(self.weights.std()),
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
            "v_mean": float(self.v_values.mean()),
            "v_std": float(self.v_values.std()),
        }
