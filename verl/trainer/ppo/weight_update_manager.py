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
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from verl import DataProto
from verl.protocol import DataProtoConfig
from verl.single_controller.base.worker import Worker
from verl.trainer.ppo.ray_trainer import Role

logger = logging.getLogger(__name__)


class WeightUpdateManager:
    """
    管理数据集权重更新的组件。
    定期遍历整个数据集，使用actor提取hidden states，
    通过reward estimator计算v值，并更新采样权重。
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        actor_worker_group,
        reward_estimator_worker_group,
        update_interval: int = 100,
        batch_size: int = 32,
        max_output_length: int = 1,
    ):
        """
        Args:
            dataset: WeightedRLHFDataset实例
            tokenizer: 分词器
            actor_worker_group: Actor worker组
            reward_estimator_worker_group: Reward estimator worker组
            update_interval: 更新间隔（步数）
            batch_size: 更新时的批次大小
            max_output_length: 提取hidden states时的最大输出长度
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.actor_worker_group = actor_worker_group
        self.reward_estimator_worker_group = reward_estimator_worker_group
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.max_output_length = max_output_length
        
        self.step_count = 0
        self.update_count = 0
        
    def should_update(self) -> bool:
        """判断是否需要更新权重"""
        return self.step_count % self.update_interval == 0
    
    def step(self):
        """增加步数计数"""
        self.step_count += 1
    
    def update_dataset_weights(self):
        """
        异步更新整个数据集的权重
        """
        logger.info(f"Starting weight update #{self.update_count + 1}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 不打乱，按顺序处理
            collate_fn=lambda x: x  # 返回原始数据
        )
        
        all_indices = []
        all_v_values = []
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 收集每个样本的indices和hidden states
                dataset_indices = []
                all_hidden_states = []
                
                for sample in batch:
                    # 只取prompt部分，添加一个dummy response token
                    prompt_length = len(sample["raw_prompt_ids"])
                    
                    # 构建单个样本的输入（prompt + 1个response token）
                    input_ids = torch.cat([
                        sample["input_ids"][:prompt_length],
                        torch.zeros(1, dtype=torch.long, device=sample["input_ids"].device)
                    ])
                    attention_mask = torch.cat([
                        sample["attention_mask"][:prompt_length],
                        torch.ones(1, dtype=torch.long, device=sample["attention_mask"].device)  
                    ])
                    position_ids = sample["position_ids"]
                    if position_ids.dim() == 2:
                        position_ids = torch.cat([
                            position_ids[:, :prompt_length],
                            position_ids[:, prompt_length:prompt_length+1]
                        ], dim=1)
                    else:
                        # 1D position_ids
                        if len(position_ids) > prompt_length:
                            position_ids = torch.cat([
                                position_ids[:prompt_length],
                                position_ids[prompt_length:prompt_length+1]
                            ])
                        else:
                            # 如果position_ids不够长，生成新的
                            position_ids = torch.cat([
                                position_ids[:prompt_length],
                                torch.tensor([prompt_length], dtype=torch.long, device=position_ids.device)
                            ])
                    
                    dataset_indices.append(sample["dataset_index"])
                    
                    # 构建单样本的DataProto
                    single_data_proto = DataProto.from_dict(
                        tensors={
                            "input_ids": input_ids.unsqueeze(0),  # Add batch dimension
                            "attention_mask": attention_mask.unsqueeze(0),
                            "position_ids": position_ids.unsqueeze(0) if position_ids.dim() == 1 else position_ids,
                            "responses": torch.zeros((1, 1), dtype=torch.long, device=input_ids.device),
                        },
                        meta_info={
                            "micro_batch_size": 1,
                            "temperature": 1.0,
                            "use_dynamic_bsz": False,
                            "calculate_entropy": False,
                            "enable_hidden_states": True,
                        }
                    )
                    
                    # 计算单个样本的hidden states
                    with torch.no_grad():
                        output = self.actor_worker_group.compute_log_prob(single_data_proto)
                        
                        if isinstance(output, tuple) and len(output) == 3:
                            _, _, hidden_states = output
                        else:
                            hidden_states = output.batch.get("hidden_states") if hasattr(output, 'batch') else None
                    
                    if hidden_states is None:
                        logger.warning(f"Sample in batch {batch_idx}: hidden_states is None, skipping")
                        dataset_indices.pop()  # Remove the last added index
                        continue
                    
                    all_hidden_states.append(hidden_states)
                
                # Skip if no valid samples
                if not all_hidden_states:
                    logger.warning(f"Batch {batch_idx}: No valid samples after processing")
                    continue
                
                # Concatenate all hidden states
                batch_hidden_states = torch.cat(all_hidden_states, dim=0)
                
                # 构建reward estimator的输入
                estimator_data = DataProto.from_dict(
                    tensors={"hidden_states": batch_hidden_states}
                )
                
                # 使用reward estimator计算v值
                reward_output = self.reward_estimator_worker_group.compute_estimated_reward(
                    estimator_data
                )
                
                # 提取v值（estimated_rewards）
                v_values = reward_output.batch["estimated_rewards"].cpu().numpy()
                
                # 收集结果 - 确保indices是数值类型
                if isinstance(dataset_indices, list) and dataset_indices:
                    # 如果是list of numpy scalars or integers
                    dataset_indices_list = [int(idx) if hasattr(idx, 'item') else int(idx) for idx in dataset_indices]
                else:
                    dataset_indices_list = dataset_indices
                all_indices.extend(dataset_indices_list)
                all_v_values.extend(v_values)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {(batch_idx + 1) * self.batch_size} samples")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        # 更新数据集权重
        if all_indices:
            # 确保indices是int类型的numpy array
            indices_array = np.array(all_indices, dtype=np.int64)
            v_values_array = np.array(all_v_values, dtype=np.float32)
            
            self.dataset.update_weights_from_v(indices_array, v_values_array)
            
            stats = self.dataset.get_weight_stats()
            logger.info(f"Weight update #{self.update_count + 1} completed. "
                       f"Updated {len(all_indices)} samples. Stats: {stats}")
            
            self.update_count += 1
        else:
            logger.warning("No samples were successfully processed in weight update")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "step_count": self.step_count,
            "update_count": self.update_count,
            "next_update_in": self.update_interval - (self.step_count % self.update_interval),
        }
        stats.update(self.dataset.get_weight_stats())
        return stats


class SyncWeightUpdateManager(WeightUpdateManager):
    """
    同步版本的权重更新管理器
    """
    
    def update_dataset_weights(self):
        """
        同步更新整个数据集的权重
        """
        logger.info(f"Starting weight update #{self.update_count + 1}")
        
        # 创建数据加载器
        from verl.utils.dataset.rl_dataset import collate_fn
        
        # 获取worker group的world_size以确定是否需要drop_last
        world_size = getattr(self.actor_worker_group, 'world_size', 1)
        drop_last = False  # 不丢弃最后的batch，我们会手动padding
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=drop_last
        )
        
        all_indices = []
        all_v_values = []
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 提取dataset indices
                dataset_indices = batch.pop("dataset_index", None)
                if dataset_indices is None:
                    logger.warning(f"Batch {batch_idx}: No dataset_index found")
                    continue
                
                # 确保dataset_indices是torch tensor
                if not isinstance(dataset_indices, torch.Tensor):
                    if isinstance(dataset_indices, np.ndarray):
                        # 如果是object dtype的numpy array，先转换为int64
                        if dataset_indices.dtype == np.object_ or dataset_indices.dtype.kind == 'O':
                            dataset_indices = dataset_indices.astype(np.int64)
                        dataset_indices = torch.from_numpy(dataset_indices).to('cpu')
                    else:
                        dataset_indices = torch.tensor(dataset_indices, dtype=torch.long, device='cpu')
                else:
                    # 如果已经是tensor，确保在CPU上
                    dataset_indices = dataset_indices.cpu()
                
                # 只使用prompt部分
                # 截断responses为1个token
                batch_size = batch["input_ids"].size(0)
                
                # 检查是否需要padding以满足并行度要求
                # 获取worker group的world_size (GPU数量)
                world_size = getattr(self.actor_worker_group, 'world_size', 1)
                if world_size > 1 and batch_size % world_size != 0:
                    # 计算需要padding的数量
                    padding_size = world_size - (batch_size % world_size)
                    logger.info(f"Batch size {batch_size} not divisible by world_size {world_size}, padding {padding_size} samples")
                    
                    # Padding所有tensor到可以被world_size整除，并移除非tensor数据
                    keys_to_remove = []
                    for key in list(batch.keys()):  # 使用list()创建副本，避免在迭代时修改
                        if isinstance(batch[key], torch.Tensor):
                            pad_shape = list(batch[key].shape)
                            pad_shape[0] = padding_size
                            # 使用第一个样本作为padding（或者使用zeros）
                            padding = batch[key][:1].repeat(padding_size, *([1] * (len(pad_shape) - 1)))
                            batch[key] = torch.cat([batch[key], padding], dim=0)
                        else:
                            # 记录非tensor的key，稍后移除（它们不需要传给actor）
                            keys_to_remove.append(key)
                    
                    # 移除非tensor数据，避免batch size不匹配
                    for key in keys_to_remove:
                        batch.pop(key, None)
                    
                    # 更新batch_size
                    padded_batch_size = batch_size + padding_size
                    # 记录原始indices的数量，padding的indices设为-1
                    original_indices = dataset_indices
                    # 使用torch.long作为dtype，这是indices的标准类型，并确保在CPU上
                    padding_indices = torch.full((padding_size,), -1, dtype=torch.long, device='cpu')
                    dataset_indices = torch.cat([dataset_indices, padding_indices])
                else:
                    padded_batch_size = batch_size
                    original_indices = dataset_indices
                    # 即使不需要padding，也移除非tensor数据
                    keys_to_remove = []
                    for key in list(batch.keys()):
                        if not isinstance(batch[key], torch.Tensor):
                            keys_to_remove.append(key)
                    for key in keys_to_remove:
                        batch.pop(key, None)
                
                batch["responses"] = batch["responses"][:, :1] if "responses" in batch else torch.zeros((padded_batch_size, 1), dtype=torch.long)
                
                # 过滤出只包含tensor的dict
                tensor_dict = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        tensor_dict[key] = value
                
                # 构建DataProto - 使用from_dict方法，只传递tensor数据
                # 启用auto_padding以支持batch_size不能被world_size整除的情况
                data_proto = DataProto.from_dict(
                    tensors=tensor_dict,
                    meta_info={
                        "micro_batch_size": padded_batch_size,
                        "temperature": 1.0,
                        "use_dynamic_bsz": False,
                        "calculate_entropy": False,  # 不需要计算entropy
                        "enable_hidden_states": True,  # 需要hidden states
                        DataProtoConfig.auto_padding_key: True,  # 启用auto_padding
                    }
                )
                
                # 使用actor计算hidden states
                with torch.no_grad():
                    output = self.actor_worker_group.compute_log_prob(data_proto)
                    
                    # 处理返回值 - 可能是tuple或DataProto
                    if isinstance(output, tuple):
                        if len(output) >= 3:
                            log_probs, entropy, hidden_states = output[:3]
                        else:
                            hidden_states = None
                    elif hasattr(output, 'batch'):
                        # Output is a DataProto
                        hidden_states = output.batch.get("hidden_states")
                    else:
                        hidden_states = None
                    
                    if hidden_states is None:
                        logger.error(
                            "hidden_states is None. Please ensure:\n"
                            "1. actor_rollout_ref.actor.output_hidden_states=True in your config\n"
                            "2. actor_rollout_ref.actor.output_hidden_states_mode is set (e.g., 'prompt_last', 'prompt_mean')\n"
                            "3. The actor worker is properly configured to output hidden states"
                        )
                        continue
                
                # 构建reward estimator的输入
                # 启用auto_padding以支持batch_size不能被world_size整除的情况
                estimator_data = DataProto.from_dict(
                    tensors={"hidden_states": hidden_states},
                    meta_info={
                        DataProtoConfig.auto_padding_key: True,  # 启用auto_padding
                    }
                )
                
                # 使用reward estimator计算v值
                reward_output = self.reward_estimator_worker_group.compute_estimated_reward(
                    estimator_data
                )
                
                v_values = reward_output.batch["estimated_rewards"]
                
                # Extract epistemic uncertainty if available (from epistemic uncertainty feature)
                # Requirements: 8.3 - Incorporate epistemic uncertainty when enabled
                epistemic_uncertainty = reward_output.batch.get("epistemic_uncertainty", None)
                
                # 如果有padding，只保留原始样本的结果
                if batch_size < padded_batch_size:
                    v_values = v_values[:batch_size]
                    if epistemic_uncertainty is not None:
                        epistemic_uncertainty = epistemic_uncertainty[:batch_size]
                    dataset_indices = original_indices
                
                # 收集结果
                if isinstance(dataset_indices, torch.Tensor):
                    dataset_indices_list = dataset_indices.cpu().numpy().tolist()
                else:
                    dataset_indices_list = dataset_indices.tolist() if hasattr(dataset_indices, 'tolist') else list(dataset_indices)
                
                all_indices.extend(dataset_indices_list)
                all_v_values.extend(v_values.cpu().numpy().tolist() if isinstance(v_values, torch.Tensor) else v_values)
                
                # Collect epistemic uncertainty values if available
                if epistemic_uncertainty is not None:
                    if not hasattr(self, '_all_epistemic_uncertainty'):
                        self._all_epistemic_uncertainty = []
                    self._all_epistemic_uncertainty.extend(
                        epistemic_uncertainty.cpu().numpy().tolist() if isinstance(epistemic_uncertainty, torch.Tensor) else epistemic_uncertainty
                    )
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        # 更新数据集权重
        if all_indices:
            # 确保indices是int类型的numpy array
            indices_array = np.array(all_indices, dtype=np.int64)
            v_values_array = np.array(all_v_values, dtype=np.float32)
            
            # Pass epistemic uncertainty if available
            # Requirements: 3.1, 3.3, 8.3 - Incorporate epistemic uncertainty in sampling weights
            epistemic_uncertainty_array = None
            if hasattr(self, '_all_epistemic_uncertainty') and self._all_epistemic_uncertainty:
                epistemic_uncertainty_array = np.array(self._all_epistemic_uncertainty, dtype=np.float32)
                # Clear for next update
                self._all_epistemic_uncertainty = []
            
            self.dataset.update_weights_from_v(
                indices_array, 
                v_values_array,
                epistemic_uncertainty=epistemic_uncertainty_array
            )
            
            stats = self.dataset.get_weight_stats()
            logger.info(f"Weight update #{self.update_count + 1} completed. Stats: {stats}")
            
            self.update_count += 1
