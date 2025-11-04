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
import torch.nn as nn
from verl.single_controller.base import Worker
from verl import DataProto
from verl.utils.device import get_device_name

class RewardEstimatorWorker(Worker):
    """
    Worker for reward estimation using a linear layer.
    This worker maintains a simple linear model that estimates rewards from hidden states.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device_name = get_device_name()
        
    def init_model(self):
        """在 GPU worker 上初始化模型"""
        hidden_size = self.config.get("hidden_size", 768)
        self.lambda_reg = self.config.get("lambda_reg", 1e-3)  # Ridge regularization
        self.use_analytical = self.config.get("use_analytical", True)  # 是否使用解析解
        
        self.model = nn.Linear(hidden_size, 1, bias=False).to(self.device_name)
        
        if self.use_analytical:
            # 初始化累积统计量用于解析解
            self.XTX = torch.zeros(hidden_size, hidden_size, device=self.device_name)
            self.XTy = torch.zeros(hidden_size, device=self.device_name)
            self.n_samples = 0
            print(f"RewardEstimatorWorker initialized on {self.device_name}, "
                  f"hidden_size={hidden_size}, lambda_reg={self.lambda_reg}, mode=analytical")
        else:
            # 使用 SGD 方法
            learning_rate = self.config.get("learning_rate", 1e-4)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            print(f"RewardEstimatorWorker initialized on {self.device_name}, "
                  f"hidden_size={hidden_size}, lr={learning_rate}, mode=SGD")
        
    def compute_estimated_reward(self, data: DataProto) -> DataProto:
        """计算估计的奖励
        
        Args:
            data: DataProto with batch["hidden_states"] of shape (batch_size, hidden_size)
        
        Returns:
            DataProto with batch["estimated_rewards"] of shape (batch_size,)
        """
        hidden_states = data.batch["hidden_states"]  # (batch_size, hidden_size)
        
        with torch.no_grad():
            self.model.eval()
            estimated_rewards = self.model(hidden_states).squeeze(-1)  # (batch_size,)
        
        output = DataProto()
        output.batch["estimated_rewards"] = estimated_rewards
        return output
    
    def update_estimator(self, data: DataProto) -> DataProto:
        """更新估计器
        
        Args:
            data: DataProto with:
                - batch["hidden_states"]: (batch_size, hidden_size)
                - batch["token_level_rewards"]: (batch_size, seq_len) or (batch_size,)
        
        Returns:
            DataProto with metrics
        """
        hidden_states = data.batch["hidden_states"]  # (batch_size, hidden_size)
        target_rewards = data.batch["token_level_rewards"]  # (batch_size, max_seq_len)
        
        # Sum rewards across sequence if needed
        if len(target_rewards.shape) == 2:
            target_rewards = target_rewards.sum(dim=-1)  # (batch_size,)
        
        if self.use_analytical:
            # 使用解析解方法
            # 累积统计量
            self.XTX += hidden_states.T @ hidden_states  # (hidden_size, hidden_size)
            self.XTy += hidden_states.T @ target_rewards  # (hidden_size,)
            self.n_samples += hidden_states.shape[0]
            
            # 求解 Ridge regression: theta = (X.T @ X + lambda*I)^{-1} @ (X.T @ y)
            XTX_reg = self.XTX + self.lambda_reg * torch.eye(
                self.XTX.shape[0], device=self.device_name
            )
            theta = torch.linalg.solve(XTX_reg, self.XTy)  # (hidden_size,)
            
            # 更新模型参数
            with torch.no_grad():
                self.model.weight.copy_(theta.unsqueeze(0))
            
            # 计算当前 batch 的 loss 用于监控
            with torch.no_grad():
                estimated_rewards = self.model(hidden_states).squeeze(-1)
                loss = nn.functional.mse_loss(estimated_rewards, target_rewards)
            
            metrics = {
                "reward_estimator/loss": loss.item(),
                "reward_estimator/n_samples": self.n_samples,
                "reward_estimator/mean_estimated": estimated_rewards.mean().item(),
                "reward_estimator/mean_target": target_rewards.mean().item(),
                "reward_estimator/XTX_cond": torch.linalg.cond(XTX_reg).item(),  # 条件数
            }
        else:
            # 使用 SGD 方法
            self.model.train()
            estimated_rewards = self.model(hidden_states).squeeze(-1)
            loss = nn.functional.mse_loss(estimated_rewards, target_rewards)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            metrics = {
                "reward_estimator/loss": loss.item(),
                "reward_estimator/mean_estimated": estimated_rewards.mean().item(),
                "reward_estimator/mean_target": target_rewards.mean().item(),
            }
        
        output = DataProto()
        output.meta_info = {"metrics": metrics}
        return output
    
    def save_checkpoint(self, path):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'use_analytical': self.use_analytical,
        }
        
        if self.use_analytical:
            # 保存累积统计量
            checkpoint['XTX'] = self.XTX
            checkpoint['XTy'] = self.XTy
            checkpoint['n_samples'] = self.n_samples
        else:
            # 保存优化器状态
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        """加载检查点"""
        if path is None:
            return
            
        checkpoint = torch.load(path, map_location=self.device_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.use_analytical and 'XTX' in checkpoint:
            # 加载累积统计量
            self.XTX = checkpoint['XTX'].to(self.device_name)
            self.XTy = checkpoint['XTy'].to(self.device_name)
            self.n_samples = checkpoint['n_samples']
            print(f"Loaded analytical estimator with {self.n_samples} accumulated samples")
        elif not self.use_analytical and 'optimizer_state_dict' in checkpoint:
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded SGD estimator with optimizer state")
    
    def reset_accumulation(self):
        """重置累积的统计量（仅对解析解模式有效）"""
        if self.use_analytical:
            hidden_size = self.XTX.shape[0]
            self.XTX = torch.zeros(hidden_size, hidden_size, device=self.device_name)
            self.XTy = torch.zeros(hidden_size, device=self.device_name)
            self.n_samples = 0
            print("Reset accumulated statistics for analytical estimator")
        else:
            print("Warning: reset_accumulation only works for analytical mode")