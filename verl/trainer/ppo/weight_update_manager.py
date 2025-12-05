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
    
    async def update_dataset_weights(self):
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
                # 提取batch中的数据
                input_ids_list = []
                attention_mask_list = []
                position_ids_list = []
                dataset_indices = []
                
                for sample in batch:
                    # 只取prompt部分，设置response长度为1
                    prompt_length = len(sample["raw_prompt_ids"])
                    
                    # 截断到prompt部分
                    input_ids = sample["input_ids"][:prompt_length]
                    attention_mask = sample["attention_mask"][:prompt_length]
                    position_ids = sample["position_ids"]
                    if position_ids.dim() == 2:
                        position_ids = position_ids[:, :prompt_length]
                    else:
                        position_ids = position_ids[:prompt_length]
                    
                    input_ids_list.append(input_ids)
                    attention_mask_list.append(attention_mask)
                    position_ids_list.append(position_ids)
                    dataset_indices.append(sample["dataset_index"])
                
                # 堆叠成batch tensor
                input_ids = torch.stack(input_ids_list)
                attention_mask = torch.stack(attention_mask_list)
                
                # 处理position_ids（可能是2D或3D）
                if position_ids_list[0].dim() == 2:
                    position_ids = torch.stack(position_ids_list)
                else:
                    position_ids = torch.stack(position_ids_list)
                
                # 生成一个token的响应以获取hidden states
                responses = torch.zeros((len(batch), 1), dtype=torch.long)
                
                # 构建DataProto - 使用from_dict方法
                data_proto = DataProto.from_dict(
                    tensors={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "responses": responses,
                    },
                    meta_info={
                        "micro_batch_size": len(batch),
                        "temperature": 1.0,
                        "use_dynamic_bsz": False,
                    }
                )
                
                # 使用actor计算hidden states
                with torch.no_grad():
                    # 调用actor的compute_log_prob方法，启用hidden_states输出
                    log_probs, entropy, hidden_states = await self.actor_worker_group.async_compute_log_prob(
                        data_proto, 
                        calculate_entropy=False,
                        enable_hidden_states=True
                    )
                
                if hidden_states is None:
                    logger.warning(f"Batch {batch_idx}: hidden_states is None, skipping")
                    continue
                
                # 构建reward estimator的输入
                estimator_data = DataProto(
                    batch={"hidden_states": hidden_states}
                )
                
                # 使用reward estimator计算v值
                reward_output = await self.reward_estimator_worker_group.async_compute_estimated_reward(
                    estimator_data
                )
                
                # 提取v值（estimated_rewards）
                v_values = reward_output.batch["estimated_rewards"].cpu().numpy()
                
                # 收集结果
                all_indices.extend(dataset_indices)
                all_v_values.extend(v_values)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {(batch_idx + 1) * self.batch_size} samples")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # 更新数据集权重
        if all_indices:
            indices_array = np.array(all_indices)
            v_values_array = np.array(all_v_values)
            
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
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
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
                
                # 只使用prompt部分
                # 截断responses为1个token
                batch_size = batch["input_ids"].size(0)
                batch["responses"] = batch["responses"][:, :1] if "responses" in batch else torch.zeros((batch_size, 1), dtype=torch.long)
                
                # 将batch转换为普通dict（如果它是TensorDict）
                if hasattr(batch, "items"):
                    batch_dict = {k: v for k, v in batch.items()}
                else:
                    batch_dict = batch
                
                # 构建DataProto - 使用from_dict方法
                data_proto = DataProto.from_dict(
                    tensors=batch_dict,
                    meta_info={
                        "micro_batch_size": batch_size,
                        "temperature": 1.0,
                        "use_dynamic_bsz": False,
                    }
                )
                
                # 使用actor计算hidden states
                with torch.no_grad():
                    output = self.actor_worker_group.compute_log_prob(
                        data_proto,
                        calculate_entropy=False,
                        enable_hidden_states=True
                    )
                    
                    if len(output) == 3:
                        log_probs, entropy, hidden_states = output
                    else:
                        log_probs, entropy = output
                        # Extract hidden states from response
                        hidden_states = output.batch.get("hidden_states")
                        if hidden_states is None:
                            raise ValueError(
                                "hidden_states is None. Please ensure:\n"
                                "1. actor_rollout_ref.actor.output_hidden_states=True in your config\n"
                                "2. actor_rollout_ref.actor.output_hidden_states_mode is set (e.g., 'prompt_last', 'prompt_mean')\n"
                                "3. The actor worker is properly configured to output hidden states"
                            )
                
                if hidden_states is None:
                    logger.warning(f"Batch {batch_idx}: hidden_states is None, skipping")
                    continue
                
                # 构建reward estimator的输入
                estimator_data = DataProto(
                    batch={"hidden_states": hidden_states}
                )
                
                # 使用reward estimator计算v值
                reward_output = self.reward_estimator_worker_group.compute_estimated_reward(
                    estimator_data
                )
                
                # 提取v值
                v_values = reward_output.batch["estimated_rewards"].cpu().numpy()
                
                # 收集结果
                all_indices.extend(dataset_indices.tolist())
                all_v_values.extend(v_values)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # 更新数据集权重
        if all_indices:
            self.dataset.update_weights_from_v(
                np.array(all_indices),
                np.array(all_v_values)
            )
            
            stats = self.dataset.get_weight_stats()
            logger.info(f"Weight update #{self.update_count + 1} completed. Stats: {stats}")
            
            self.update_count += 1
