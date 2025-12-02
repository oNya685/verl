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

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, List
import logging

logger = logging.getLogger(__name__)


class WeightedBatchSampler(Sampler):
    """
    批次采样器，支持有放回的加权采样。
    每个batch通过权重概率采样生成。
    """
    
    def __init__(self, dataset, batch_size: int, num_batches: int = None, seed: int = None):
        """
        Args:
            dataset: WeightedRLHFDataset实例
            batch_size: 批次大小
            num_batches: 要生成的批次数量，如果为None则持续生成
            seed: 随机种子
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0
        
        # 初始化随机数生成器
        self.rng = np.random.default_rng(seed)
        
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次索引"""
        # 设置随机种子以保证可重复性
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed + self.epoch)
        
        # 获取当前的采样权重
        weights = self.dataset.get_sampling_weights()
        dataset_size = len(self.dataset)
        
        # 生成批次
        batch_count = 0
        while True:
            # 有放回采样
            batch_indices = self.rng.choice(
                dataset_size, 
                size=self.batch_size,
                p=weights,
                replace=True  # 有放回采样
            )
            
            yield batch_indices.tolist()
            
            batch_count += 1
            if self.num_batches is not None and batch_count >= self.num_batches:
                break
    
    def __len__(self):
        """返回采样器的长度（批次数）"""
        if self.num_batches is not None:
            return self.num_batches
        else:
            # 如果没有指定批次数，返回一个默认值
            return len(self.dataset) // self.batch_size
    
    def set_epoch(self, epoch: int):
        """设置epoch，用于改变随机种子"""
        self.epoch = epoch


class DynamicWeightedSampler(Sampler):
    """
    动态加权采样器，支持在训练过程中更新权重。
    """
    
    def __init__(
        self, 
        dataset, 
        batch_size: int,
        update_interval: int = 100,
        seed: int = None
    ):
        """
        Args:
            dataset: WeightedRLHFDataset实例
            batch_size: 批次大小
            update_interval: 权重更新间隔（批次数）
            seed: 随机种子
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.seed = seed
        
        self.current_batch = 0
        self.pending_weight_updates = []
        
        # 初始化随机数生成器
        self.rng = np.random.default_rng(seed)
        
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次索引"""
        dataset_size = len(self.dataset)
        
        while True:
            # 每隔一定批次更新权重
            if self.current_batch % self.update_interval == 0:
                self._apply_pending_updates()
            
            # 获取当前权重
            weights = self.dataset.get_sampling_weights()
            
            # 有放回采样
            batch_indices = self.rng.choice(
                dataset_size,
                size=self.batch_size,
                p=weights,
                replace=True
            )
            
            self.current_batch += 1
            yield batch_indices.tolist()
    
    def __len__(self):
        """返回一个epoch的批次数"""
        return len(self.dataset) // self.batch_size
    
    def add_weight_update(self, indices: np.ndarray, v_values: np.ndarray):
        """
        添加待处理的权重更新
        
        Args:
            indices: 样本索引
            v_values: 对应的v值（reward estimator输出）
        """
        self.pending_weight_updates.append((indices, v_values))
    
    def _apply_pending_updates(self):
        """应用所有待处理的权重更新"""
        if not self.pending_weight_updates:
            return
        
        # 合并所有更新
        all_indices = []
        all_v_values = []
        
        for indices, v_values in self.pending_weight_updates:
            all_indices.extend(indices)
            all_v_values.extend(v_values)
        
        # 应用更新
        if all_indices:
            self.dataset.update_weights_from_v(
                np.array(all_indices),
                np.array(all_v_values)
            )
            
            stats = self.dataset.get_weight_stats()
            logger.info(f"Applied weight updates for {len(all_indices)} samples. Stats: {stats}")
        
        # 清空待处理列表
        self.pending_weight_updates.clear()
