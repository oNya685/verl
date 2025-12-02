#!/usr/bin/env python3
"""
测试权重采样配置是否正确加载
"""
import sys
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import os

def test_config():
    config_dir = os.path.join(os.path.dirname(__file__), "verl/trainer/config")
    
    # 初始化Hydra配置
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # 尝试加载配置并覆盖weighted sampling参数
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "data.enable_weighted_sampling=true",
                "data.weight_update_interval=50",
                "data.weight_update_batch_size=32",
            ]
        )
        
        print("Configuration loaded successfully!")
        print("\nWeighted sampling configuration:")
        print(f"  enable_weighted_sampling: {cfg.data.enable_weighted_sampling}")
        print(f"  weight_update_interval: {cfg.data.weight_update_interval}")
        print(f"  weight_update_batch_size: {cfg.data.weight_update_batch_size}")
        print(f"  weight_eps: {cfg.data.weight_eps}")
        print(f"  initial_v: {cfg.data.initial_v}")
        print(f"  min_weight: {cfg.data.min_weight}")
        
        # 打印完整的data配置结构
        print("\nFull data config:")
        print(OmegaConf.to_yaml(cfg.data, resolve=False))

if __name__ == "__main__":
    test_config()
