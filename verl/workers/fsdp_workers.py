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
"""
The main entry point to run the PPO algorithm
"""

import datetime
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys, hidden_states = self.actor.compute_log_prob(data=data, calculate_entropy=True, enable_hidden_states=True)
            if hidden_states is not None:
                output = DataProto.from_dict(
                    tensors={"old_log_probs": output, "entropys": entropys, "hidden_states": hidden_states},
                    meta_info={"temperature": self.config.rollout.temperature},
                )
            else:
                output = DataProto.from_dict(
                    tensors={"old_log_probs": output, "entropys": entropys},
                    meta_info={"temperature": self.config.rollout.temperature},
                )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            output, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_representations(self, data: DataProto):
        """Compute semantic representation (mean of last layer token embeddings) for a batch of data.
    
        This function extracts the last hidden states from the actor model and computes
        the mean pooling over the sequence dimension (excluding padding tokens).

        Data is automatically sharded across GPUs by DP_COMPUTE_PROTO.
        Each GPU receives different data and computes representations independently.

        Args:
            data: DataProto with:
                  - batch["input_ids"]: (total_batch_size, seq_len) - auto-split across GPUs
                  - batch["attention_mask"]: (total_batch_size, seq_len) - auto-split across GPUs
                  - meta_info["pooling_strategy"]: (optional) "mean", "max", "last", "first", default="mean"
                  - meta_info["micro_batch_size"]: (optional) micro batch size for processing, default=None (process all at once)

        Returns:
            DataProto with:
                  - batch["representations"]: (batch_size_per_gpu, hidden_dim)

        Example:
            # Mean pooling (default)
            data = DataProto.from_dict(
                tensors={"input_ids": input_ids, "attention_mask": attention_mask},
                meta_info={"pooling_strategy": "mean"}
            )
            output = worker_group.compute_semantic_representation(data)
            representations = output.batch["representations"]  # (batch_size, hidden_dim)
        """
        import torch

        # Move data to GPU
        data = data.to(get_device_id())
        input_ids = data.batch["input_ids"]  # (batch_size_per_gpu, seq_len)
        attention_mask = data.batch["attention_mask"]  # (batch_size_per_gpu, seq_len)
        position_ids = data.batch["position_ids"] # (batch_size_per_gpu, seq_len)

        # Get pooling strategy from meta_info
        pooling_strategy = "last" # self.config.scheduler.pooling_strategy

        # Offload handling
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Process in micro batches if specified
        batch_size = input_ids.shape[0]
        # micro_batch_size = min(self.config.scheduler.max_actor_batch_size_per_gpu,batch_size)
        micro_batch_size = min(128, batch_size)
        all_representations = []
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            micro_input_ids = input_ids[i:end_idx]
            micro_attention_mask = attention_mask[i:end_idx]
            micro_position_ids = position_ids[i:end_idx]

            micro_repr = self._compute_representation_single_batch(
                micro_input_ids, micro_attention_mask, micro_position_ids,pooling_strategy
            )
            all_representations.append(micro_repr)

        representations = torch.cat(all_representations, dim=0)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        # Offload back if needed
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_semantic_representation", logger=logger)

        # Return as DataProto
        output = DataProto.from_dict(tensors={"representations": representations})
        output = output.to("cpu")

        return output

    def _compute_representation_single_batch(self, input_ids, attention_mask, position_ids,pooling_strategy):
        """Helper function to compute representation for a single batch.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            pooling_strategy: "mean", "max", "last", "first"

        Returns:
            representations: (batch_size, hidden_dim)
        """
        import torch

        with torch.no_grad():
            self.actor_module_fsdp.eval()

            # Forward pass to get hidden states
            outputs = self.actor_module_fsdp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
                #TODO: check what is this
                use_cache=False,
            )

            # Get last hidden state: (batch_size, seq_len, hidden_dim)
            last_hidden_state = outputs.hidden_states[-1]

            if pooling_strategy == "mean":
                # Mean pooling over sequence dimension (excluding padding)
                # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()

                # Sum over sequence dimension
                sum_hidden = (last_hidden_state * attention_mask_expanded).sum(dim=1)  # (batch_size, hidden_dim)

                # Count non-padding tokens
                sum_mask = attention_mask_expanded.sum(dim=1)  # (batch_size, 1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

                # Mean pooling
                representations = sum_hidden / sum_mask  # (batch_size, hidden_dim)

            elif pooling_strategy == "last":
                # Use the last non-padding token
                # Find the last non-padding position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
                batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                representations = last_hidden_state[batch_indices, seq_lengths]  # (batch_size, hidden_dim)

            else:
                raise ValueError(f"Unknown pooling_strategy: {pooling_strategy}. Supported: 'mean', 'last'")

        return representations


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data = rm_data.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# class RewardEstimatorWorker(Worker, DistProfilerExtension):
#     """
#     Worker for reward estimation using a linear layer.
#     This worker maintains a simple linear model that estimates rewards from hidden states.
#     """
#     def __init__(self, config: FSDPCriticConfig):
#         Worker.__init__(self)
#         omega_profiler_config = config.get("profiler", {})
#         profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
#         if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
#             tool_config = omega_conf_to_dataclass(
#                 omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
#             )
#         else:
#             tool_config = None
#         DistProfilerExtension.__init__(
#             self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
#         )
#         import torch.distributed

#         self.config = config
#         print(f"[RewardEstimatorWorker] Initializing worker rank={self.rank}")
        
#         # --- 新增：从配置获取dtype ---
#         torch_dtype_str = self.config.model.fsdp_config.get("model_dtype", "fp32")
#         from verl.utils.torch_dtypes import PrecisionType
#         self.torch_dtype = PrecisionType.to_dtype(torch_dtype_str)
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Target dtype from config: {torch_dtype_str} -> {self.torch_dtype}")
#         # -----------------------------
        
#         if not torch.distributed.is_initialized():
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: torch.distributed not initialized, init_process_group...")
#             torch.distributed.init_process_group(
#                 backend=get_nccl_backend(),
#                 timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
#                 init_method=os.environ.get("DIST_INIT_METHOD", None),
#             )
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Distributed initialized, world_size={torch.distributed.get_world_size()}")
#         else:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: torch.distributed already initialized")
            
#         self.config: FSDPCriticConfig = config

#         # build device mesh for Ulysses Sequence Parallel
#         world_size = torch.distributed.get_world_size()
#         from torch.distributed.device_mesh import init_device_mesh

#         fsdp_size = self.config.model.fsdp_config.fsdp_size
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Creating device mesh: world_size={world_size}, fsdp_size={fsdp_size}")
#         self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Device mesh created: {self.device_mesh}")

#         self.ulysses_device_mesh = None
#         self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
#         dp = world_size // self.ulysses_sequence_parallel_size
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Ulysses SP config: sp_size={self.ulysses_sequence_parallel_size}, dp={dp}")
        
#         if self.ulysses_sequence_parallel_size > 1:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Initializing Ulysses device mesh...")
#             self.ulysses_device_mesh = init_device_mesh(
#                 device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
#             )
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Ulysses device mesh created: {self.ulysses_device_mesh}")
#         else:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Ulysses SP disabled (sp_size=1)")
            
#         self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

#         # create training dispatch
#         if self.ulysses_device_mesh is not None:
#             is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
#             dp_rank = self.ulysses_device_mesh["dp"].get_local_rank()
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Registering dispatch for reward_estimator: dp_rank={dp_rank}, is_collect={is_collect}")
#             self._register_dispatch_collect_info(
#                 "reward_estimator", dp_rank=dp_rank, is_collect=is_collect
#             )
#         else:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Registering dispatch for reward_estimator: dp_rank={self.rank}, is_collect=True")
#             self._register_dispatch_collect_info("reward_estimator", dp_rank=self.rank, is_collect=True)

        
#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def init_model(self):
#         """在 GPU worker 上初始化模型"""
#         print(f"[init_model] Rank {self.rank}: Starting model initialization on device {get_device_id()}")
#         hidden_size = self.config.get("hidden_size", 896)
#         self.lambda_reg = self.config.get("lambda_reg", 1e-3)  # Ridge regularization
#         self.use_analytical = self.config.get("use_analytical", True)  # 是否使用解析解
        
#         print(f"[init_model] Rank {self.rank}: Creating Linear model: in_features={hidden_size}, out_features=1, dtype={self.torch_dtype}")
#         # --- 修改：使用配置的dtype创建模型 ---
#         self.model = torch.nn.Linear(hidden_size, 1, bias=False).to(get_device_id(), dtype=self.torch_dtype)
#         print(f"[init_model] Rank {self.rank}: Model created on device {get_device_id()}, weight dtype: {self.model.weight.dtype}")
        
#         if self.use_analytical:
#             # 初始化累积统计量用于解析解
#             print(f"[init_model] Rank {self.rank}: Initializing analytical mode - accumulating XTX and XTy with dtype={self.torch_dtype}")
#             # --- 修改：统计量也使用相同的dtype ---
#             self.XTX = torch.zeros(hidden_size, hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.XTy = torch.zeros(hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.n_samples = 0
#             print(f"[init_model] Rank {self.rank}: Model initialized on {get_device_id()}, "
#                   f"hidden_size={hidden_size}, lambda_reg={self.lambda_reg}, mode=analytical, dtype={self.torch_dtype}")
#         else:
#             # 使用 SGD 方法
#             learning_rate = self.config.get("learning_rate", 1e-4)
#             print(f"[init_model] Rank {self.rank}: Initializing SGD mode - creating AdamW optimizer with lr={learning_rate}")
#             self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
#             print(f"[init_model] Rank {self.rank}: Model initialized on {get_device_id()}, "
#                   f"hidden_size={hidden_size}, lr={learning_rate}, mode=SGD, dtype={self.torch_dtype}")
        
#         # 打印模型权重信息
#         print(f"[init_model] Rank {self.rank}: Model weight shape: {self.model.weight.shape}, dtype: {self.model.weight.dtype}")
        
#     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_estimator"))
#     def compute_estimated_reward(self, data: DataProto) -> DataProto:
#         """计算估计的奖励
        
#         Args:
#             data: DataProto with batch["hidden_states"] of shape (batch_size, hidden_size)
        
#         Returns:
#             DataProto with batch["estimated_rewards"] of shape (batch_size,)
#         """
#         data = data.to(get_device_id())
#         print(f"[compute_estimated_reward] Rank {self.rank}: === Starting reward estimation ===")
#         hidden_states = data.batch["representations"]  # (batch_size, hidden_size)
#         print(f"[compute_estimated_reward] Rank {self.rank}: Input hidden_states shape: {hidden_states.shape}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Input hidden_states dtype: {hidden_states.dtype}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Model weight dtype: {self.model.weight.dtype}")
        
#         # --- 新增：dtype一致性检查 ---
#         if hidden_states.dtype != self.model.weight.dtype:
#             print(f"[compute_estimated_reward] Rank {self.rank}: WARNING! dtype mismatch detected! "
#                   f"hidden_states={hidden_states.dtype}, model.weight={self.model.weight.dtype}")
#             print(f"[compute_estimated_reward] Rank {self.rank}: Attempting to cast hidden_states to {self.model.weight.dtype}")
#             hidden_states = hidden_states.to(self.model.weight.dtype)
        
#         with torch.no_grad():
#             self.model.eval()
#             print(f"[compute_estimated_reward] Rank {self.rank}: Model in eval mode")
#             estimated_rewards = self.model(hidden_states).squeeze(-1)  # (batch_size,)
        
#         print(f"[compute_estimated_reward] Rank {self.rank}: Output estimated_rewards shape: {estimated_rewards.shape}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Estimated rewards (first 5): {estimated_rewards[:5] if estimated_rewards.numel() > 0 else 'N/A'}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Estimated rewards mean: {estimated_rewards.mean().item():.6f}, std: {estimated_rewards.std().item():.6f}")
        
#         output = DataProto.from_dict(tensors={"estimated_rewards": estimated_rewards})
#         output = output.to("cpu")
#         print(f"[compute_estimated_reward] Rank {self.rank}: === Reward estimation completed ===")
#         return output
    
#     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_estimator"))
#     def update_estimator(self, data: DataProto) -> DataProto:
#         """更新估计器
        
#         Args:
#             data: DataProto with:
#                 - batch["hidden_states"]: (batch_size, hidden_size)
#                 - batch["token_level_rewards"]: (batch_size, seq_len) or (batch_size,)
        
#         Returns:
#             DataProto with metrics
#         """
#         data = data.to(get_device_id())
#         print(f"[update_estimator] Rank {self.rank}: === Starting estimator update ===")
#         hidden_states = data.batch["representations"]  # (batch_size, hidden_size)
#         target_rewards = data.batch["token_level_rewards"]  # (batch_size, max_seq_len)
        
#         print(f"[update_estimator] Rank {self.rank}: hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
#         print(f"[update_estimator] Rank {self.rank}: target_rewards shape: {target_rewards.shape}, dtype: {target_rewards.dtype}")
        
#         # --- 新增：dtype一致性处理 ---
#         if hidden_states.dtype != self.model.weight.dtype:
#             print(f"[update_estimator] Rank {self.rank}: Casting hidden_states from {hidden_states.dtype} to {self.model.weight.dtype}")
#             hidden_states = hidden_states.to(self.model.weight.dtype)
        
#         # Sum rewards across sequence if needed
#         if len(target_rewards.shape) == 2:
#             print(f"[update_estimator] Rank {self.rank}: Summing token-level rewards across seq_len dimension")
#             target_rewards = target_rewards.sum(dim=-1)  # (batch_size,)
#             print(f"[update_estimator] Rank {self.rank}: Summed target_rewards shape: {target_rewards.shape}")

#         print(f"[update_estimator] Rank {self.rank}: mode={ 'analytical' if self.use_analytical else 'SGD'}")

#         if self.use_analytical:
#             # 使用解析解方法
#             print(f"[update_estimator] Rank {self.rank}: --- Analytical update start ---")
#             print(f"[update_estimator] Rank {self.rank}: Accumulating statistics: n_samples before={self.n_samples}")
            
#             # 累积统计量
#             self.XTX += hidden_states.T @ hidden_states  # (hidden_size, hidden_size)
#             self.XTy += hidden_states.T @ target_rewards  # (hidden_size,)
#             self.n_samples += hidden_states.shape[0]
            
#             print(f"[update_estimator] Rank {self.rank}: n_samples after={self.n_samples}, batch size={hidden_states.shape[0]}")
#             print(f"[update_estimator] Rank {self.rank}: XTX shape: {self.XTX.shape}, XTX norm: {self.XTX.norm().item():.4f}")
#             print(f"[update_estimator] Rank {self.rank}: XTy shape: {self.XTy.shape}, XTy norm: {self.XTy.norm().item():.4f}")
            
#             # 求解 Ridge regression: theta = (X.T @ X + lambda*I)^{-1} @ (X.T @ y)
#             print(f"[update_estimator] Rank {self.rank}: Solving Ridge regression with lambda_reg={self.lambda_reg}")
#             XTX_reg = self.XTX + self.lambda_reg * torch.eye(
#                 self.XTX.shape[0], device=get_device_id(), dtype=self.torch_dtype
#             )
#             theta = torch.linalg.solve(XTX_reg, self.XTy)  # (hidden_size,)
            
#             # 计算条件数
#             condition_number = torch.linalg.cond(XTX_reg).item()
#             print(f"[update_estimator] Rank {self.rank}: XTX_reg condition number: {condition_number:.4f}")
            
#             # 更新模型参数
#             with torch.no_grad():
#                 self.model.weight.copy_(theta.unsqueeze(0))
            
#             print(f"[update_estimator] Rank {self.rank}: Model weight updated, weight norm: {theta.norm().item():.4f}")
            
#             # 计算当前 batch 的 loss 用于监控
#             with torch.no_grad():
#                 estimated_rewards = self.model(hidden_states).squeeze(-1)
#                 loss = torch.nn.functional.mse_loss(estimated_rewards, target_rewards)
            
#             metrics = {
#                 "reward_estimator/loss": loss.item(),
#                 "reward_estimator/n_samples": self.n_samples,
#                 "reward_estimator/mean_estimated": estimated_rewards.mean().item(),
#                 "reward_estimator/mean_target": target_rewards.mean().item(),
#                 "reward_estimator/XTX_cond": torch.linalg.cond(XTX_reg).item(),  # 条件数
#             }
#             print(f"[update_estimator] Rank {self.rank}: Batch loss: {loss.item():.6f}")
#             print(f"[update_estimator] Rank {self.rank}: Mean estimated: {estimated_rewards.mean().item():.6f}, Mean target: {target_rewards.mean().item():.6f}")
#         else:
#             # 使用 SGD 方法
#             print(f"[update_estimator] Rank {self.rank}: --- SGD update start ---")
#             self.model.train()
#             estimated_rewards = self.model(hidden_states).squeeze(-1)
#             loss = torch.nn.functional.mse_loss(estimated_rewards, target_rewards)
            
#             print(f"[update_estimator] Rank {self.rank}: Forward pass completed, loss: {loss.item():.6f}")
#             print(f"[update_estimator] Rank {self.rank}: Mean estimated: {estimated_rewards.mean().item():.6f}, Mean target: {target_rewards.mean().item():.6f}")
            
#             self.optimizer.zero_grad()
#             loss.backward()
#             print(f"[update_estimator] Rank {self.rank}: Backward pass completed")
            
#             # 打印梯度信息
#             grad_norm = self.model.weight.grad.norm().item()
#             print(f"[update_estimator] Rank {self.rank}: Gradient norm: {grad_norm:.4f}")
            
#             self.optimizer.step()
#             print(f"[update_estimator] Rank {self.rank}: Optimizer step completed")
            
#             metrics = {
#                 "reward_estimator/loss": loss.item(),
#                 "reward_estimator/mean_estimated": estimated_rewards.mean().item(),
#                 "reward_estimator/mean_target": target_rewards.mean().item(),
#             }
        
#         output = DataProto(meta_info={"metrics": metrics})
#         output = output.to('cpu')
#         print(f"[update_estimator] Rank {self.rank}: Final metrics: {metrics}")
#         print(f"[update_estimator] Rank {self.rank}: === Estimator update completed ===")
#         return output
    
#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def save_checkpoint(self, path):
#         """保存检查点"""
#         print(f"[save_checkpoint] Rank {self.rank}: Saving checkpoint to {path}")
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'use_analytical': self.use_analytical,
#             'torch_dtype': self.torch_dtype,  # 保存dtype信息
#         }
        
#         if self.use_analytical:
#             # 保存累积统计量
#             print(f"[save_checkpoint] Rank {self.rank}: Saving analytical mode stats (n_samples={self.n_samples})")
#             checkpoint['XTX'] = self.XTX
#             checkpoint['XTy'] = self.XTy
#             checkpoint['n_samples'] = self.n_samples
#         else:
#             # 保存优化器状态
#             print(f"[save_checkpoint] Rank {self.rank}: Saving SGD mode optimizer state")
#             checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
#         torch.save(checkpoint, path)
#         print(f"[save_checkpoint] Rank {self.rank}: Checkpoint saved successfully")
        
#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def load_checkpoint(self, path):
#         """加载检查点"""
#         if path is None:
#             print(f"[load_checkpoint] Rank {self.rank}: Path is None, skipping checkpoint load")
#             return
            
#         print(f"[load_checkpoint] Rank {self.rank}: Loading checkpoint from {path}")
#         checkpoint = torch.load(path, map_location=get_device_id())
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"[load_checkpoint] Rank {self.rank}: Model state dict loaded")
        
#         # --- 新增：加载时检查dtype ---
#         if 'torch_dtype' in checkpoint:
#             loaded_dtype = checkpoint['torch_dtype']
#             print(f"[load_checkpoint] Rank {self.rank}: Checkpoint dtype: {loaded_dtype}")
#             if loaded_dtype != self.torch_dtype:
#                 print(f"[load_checkpoint] Rank {self.rank}: WARNING: Checkpoint dtype {loaded_dtype} != current config dtype {self.torch_dtype}")
        
#         if self.use_analytical and 'XTX' in checkpoint:
#             # 加载累积统计量
#             self.XTX = checkpoint['XTX'].to(get_device_id())
#             self.XTy = checkpoint['XTy'].to(get_device_id())
#             self.n_samples = checkpoint['n_samples']
#             print(f"[load_checkpoint] Rank {self.rank}: Loaded analytical estimator with {self.n_samples} accumulated samples")
#         elif not self.use_analytical and 'optimizer_state_dict' in checkpoint:
#             # 加载优化器状态
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             print(f"[load_checkpoint] Rank {self.rank}: Loaded SGD estimator with optimizer state")
#         else:
#             print(f"[load_checkpoint] Rank {self.rank}: No appropriate state found for current mode")
#         print(f"[load_checkpoint] Rank {self.rank}: Checkpoint loading completed")
    
#     def reset_accumulation(self):
#         """重置累积的统计量（仅对解析解模式有效）"""
#         print(f"[reset_accumulation] Rank {self.rank}: Reset request received")
#         if self.use_analytical:
#             hidden_size = self.XTX.shape[0]
#             self.XTX = torch.zeros(hidden_size, hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.XTy = torch.zeros(hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.n_samples = 0
#             print(f"[reset_accumulation] Rank {self.rank}: Reset accumulated statistics for analytical estimator")
#         else:
#             print(f"[reset_accumulation] Rank {self.rank}: Warning: reset_accumulation only works for analytical mode (current mode: SGD)")

# import torch.nn as nn
# from typing import Optional

# class RewardModel(nn.Module):
#     """
#     一个统一的奖励模型，支持多种可配置的架构。

#     Args:
#         input_size (int): 输入 hidden_states 的维度。
#         config (dict): 包含模型配置的字典，例如:
#             - estimator_arch (str): 'mlp_robust', 'linear_sgd', 'mlp_sigmoid_bce'
#             - hidden_layer_size (int): MLP 的隐藏层大小。
#             - dropout_rate (float): Dropout 比例。
#             - dtype (torch.dtype): 模型的数据类型。
#     """
#     def __init__(self, input_size: int, config: dict):
#         super().__init__()
#         self.config = config
#         self.input_size = input_size
#         self.arch = "mlp_robust"
#         self.dtype = self.config.get("dtype", torch.float32)
        
#         # 根据架构名称构建网络
#         if self.arch == 'mlp_robust':
#             self.network = self._build_mlp_robust()
#         elif self.arch == 'linear_sgd':
#             self.network = self._build_linear_sgd()
#         elif self.arch == 'mlp_sigmoid_bce':
#             self.network = self._build_mlp_sigmoid_bce()
#         else:
#             raise ValueError(f"Unsupported reward model architecture: {self.arch}")

#         # 权重初始化
#         self._initialize_weights()

#     def _build_mlp_robust(self) -> nn.Module:
#         """
#         方案 A (推荐): 鲁棒的 MLP，无输出激活函数。
#         预测一个无界的值，依赖损失函数来拟合目标范围。
#         这是最直接且稳定的回归方法。
#         """
#         hidden_size = self.config.get("hidden_layer_size", self.input_size)
#         dropout_rate = self.config.get("dropout_rate", 0.1)
        
#         return nn.Sequential(
#             nn.Linear(self.input_size, hidden_size, dtype=self.dtype),
#             nn.LayerNorm(hidden_size, dtype=self.dtype), # 新增: LayerNorm 稳定训练
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_size, 1, dtype=self.dtype)
#         )

#     def _build_linear_sgd(self) -> nn.Module:
#         """
#         方案 B: 简单的线性模型。
#         替代解析解，通过 SGD/Adam 进行迭代优化，计算成本低。
#         """
#         return nn.Linear(self.input_size, 1, bias=False, dtype=self.dtype)

#     def _build_mlp_sigmoid_bce(self) -> nn.Module:
#         """
#         方案 C: 带 Sigmoid 输出的 MLP。
#         设计用于配合 BCELoss，将奖励视为一个概率。
#         """
#         hidden_size = self.config.get("hidden_layer_size", self.input_size)
#         dropout_rate = self.config.get("dropout_rate", 0.1)
        
#         return nn.Sequential(
#             nn.Linear(self.input_size, hidden_size, dtype=self.dtype),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_size, 1, dtype=self.dtype),
#             nn.Sigmoid()
#         )
    
#     def _initialize_weights(self):
#         """保守初始化：小方差，避免对初始噪声过拟合"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.normal_(module.weight, mean=0.0, std=0.01)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.network(x).squeeze(-1)

# class RewardEstimatorWorker(Worker, DistProfilerExtension):
#     """
#     Worker for reward estimation using a neural network.
#     This worker maintains a neural network model that estimates rewards from hidden states,
#     with output scaled to [0,1] using Sigmoid activation.
#     """
#     def __init__(self, config: FSDPCriticConfig):
#         Worker.__init__(self)
#         omega_profiler_config = config.get("profiler", {})
#         profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
#         if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
#             tool_config = omega_conf_to_dataclass(
#                 omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
#             )
#         else:
#             tool_config = None
#         DistProfilerExtension.__init__(
#             self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
#         )
#         import torch.distributed

#         self.config = config
        
#         # --- 训练策略配置 ---
#         self.tracks_noisy_labels = True  # 明确模式
#         self.fixed_learning_rate = self.config.get("learning_rate", 1e-3)  # 提高默认值适应噪声
#         self.gradient_clip_norm = self.config.get("gradient_clip_norm", 1.0)  # 新增：梯度裁剪阈值
#         self.weight_decay = self.config.get("weight_decay", 1e-4)  # 新增：权重衰减
#         # -----------------------------
        
#         torch_dtype_str = self.config.model.fsdp_config.get("model_dtype", "fp32")
#         from verl.utils.torch_dtypes import PrecisionType
#         self.torch_dtype = PrecisionType.to_dtype(torch_dtype_str)
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Target dtype from config: {torch_dtype_str} -> {self.torch_dtype}")
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: TRAINING MODE: Tracking NOISY labels with fixed_lr={self.fixed_learning_rate}, grad_clip={self.gradient_clip_norm}")
#         print(f"[RewardEstimatorWorker] Initializing worker rank={self.rank}")
#         if not torch.distributed.is_initialized():
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: torch.distributed not initialized, init_process_group...")
#             torch.distributed.init_process_group(
#                 backend=get_nccl_backend(),
#                 timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
#                 init_method=os.environ.get("DIST_INIT_METHOD", None),
#             )
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Distributed initialized, world_size={torch.distributed.get_world_size()}")
#         else:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: torch.distributed already initialized")
            
#         self.config: FSDPCriticConfig = config

#         # build device mesh for Ulysses Sequence Parallel
#         world_size = torch.distributed.get_world_size()
#         from torch.distributed.device_mesh import init_device_mesh

#         fsdp_size = self.config.model.fsdp_config.fsdp_size
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Creating device mesh: world_size={world_size}, fsdp_size={fsdp_size}")
#         self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Device mesh created: {self.device_mesh}")

#         self.ulysses_device_mesh = None
#         self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
#         dp = world_size // self.ulysses_sequence_parallel_size
#         print(f"[RewardEstimatorWorker] Rank {self.rank}: Ulysses SP config: sp_size={self.ulysses_sequence_parallel_size}, dp={dp}")
        
#         if self.ulysses_sequence_parallel_size > 1:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Initializing Ulysses device mesh...")
#             self.ulysses_device_mesh = init_device_mesh(
#                 device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
#             )
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Ulysses device mesh created: {self.ulysses_device_mesh}")
#         else:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Ulysses SP disabled (sp_size=1)")
            
#         self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

#         # create training dispatch
#         if self.ulysses_device_mesh is not None:
#             is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
#             dp_rank = self.ulysses_device_mesh["dp"].get_local_rank()
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Registering dispatch for reward_estimator: dp_rank={dp_rank}, is_collect={is_collect}")
#             self._register_dispatch_collect_info(
#                 "reward_estimator", dp_rank=dp_rank, is_collect=is_collect
#             )
#         else:
#             print(f"[RewardEstimatorWorker] Rank {self.rank}: Registering dispatch for reward_estimator: dp_rank={self.rank}, is_collect=True")
#             self._register_dispatch_collect_info("reward_estimator", dp_rank=self.rank, is_collect=True)
        
#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def init_model(self):
#         """在 GPU worker 上初始化模型"""
#         print(f"[init_model] Rank {self.rank}: Starting model initialization on device {get_device_id()}")
#         hidden_size = self.config.get("hidden_size", 896)
#         self.use_analytical = self.config.get("use_analytical", False) # 默认改为 False

#         if self.use_analytical:
#             print(f"[init_model] Rank {self.rank}: ANALYTICAL mode - Linear regression")
#             self.lambda_reg = self.config.get("lambda_reg", 1e-3)
#             self.model = torch.nn.Linear(hidden_size, 1, bias=False).to(get_device_id(), dtype=self.torch_dtype)
#             self.XTX = torch.zeros(hidden_size, hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.XTy = torch.zeros(hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.n_samples = 0
#         else:
#             # --- 神经网络配置 ---
#             self.estimator_arch = "mlp_robust" # 可配置架构
#             model_config = {
#                 "estimator_arch": self.estimator_arch,
#                 "hidden_layer_size": self.config.get("hidden_layer_size", hidden_size),
#                 "dropout_rate": self.config.get("dropout_rate", 0.15),
#                 "dtype": self.torch_dtype,
#             }
            
#             print(f"[init_model] Rank {self.rank}: NEURAL NET mode")
#             print(f"[init_model] Rank {self.rank}: Architecture: {self.estimator_arch}")

#             self.model = RewardModel(input_size=hidden_size, config=model_config).to(get_device_id())

#             # 根据架构选择损失函数
#             if self.estimator_arch == 'mlp_sigmoid_bce':
#                 self.loss_fn = nn.BCELoss()
#                 print("[init_model] Rank {self.rank}: Using BCELoss (Binary Cross-Entropy)")
#             else: # for 'mlp_robust' and 'linear_sgd'
#                 self.loss_fn = nn.MSELoss() # 或者 nn.SmoothL1Loss()
#                 print(f"[init_model] Rank {self.rank}: Using MSELoss")
                
#             self.optimizer = torch.optim.AdamW(
#                 self.model.parameters(),
#                 lr=self.fixed_learning_rate,
#                 weight_decay=self.weight_decay,
#             )
            
#             # 推荐：添加学习率调度器来稳定训练后期
#             self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)
#             print(f"[init_model] Rank {self.rank}: LEARNING RATE SCHEDULER: ENABLED (CosineAnnealingLR)")

#         total_params = sum(p.numel() for p in self.model.parameters())
#         print(f"[init_model] Rank {self.rank}: Model ready: {total_params:,} params, dtype={self.torch_dtype}")
#         print(f"[init_model] Rank {self.rank}: Architecture details:\n{self.model}")

              
#     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_estimator"))
#     def compute_estimated_reward(self, data: DataProto) -> DataProto:
#         """计算估计的奖励
        
#         Args:
#             data: DataProto with batch["hidden_states"] of shape (batch_size, hidden_size)
        
#         Returns:
#             DataProto with batch["estimated_rewards"] of shape (batch_size,)
#             输出值已通过 Sigmoid 缩放到 [0, 1]
#         """
#         data = data.to(get_device_id())
#         print(f"[compute_estimated_reward] Rank {self.rank}: === Starting reward estimation ===")
#         hidden_states = data.batch["hidden_states"]  # (batch_size, hidden_size)
#         print(f"[compute_estimated_reward] Rank {self.rank}: Input hidden_states shape: {hidden_states.shape}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Input hidden_states dtype: {hidden_states.dtype}")
        
#         if hidden_states.dtype != self.torch_dtype:
#             print(f"[compute_estimated_reward] Rank {self.rank}: Casting hidden_states from {hidden_states.dtype} to {self.torch_dtype}")
#             hidden_states = hidden_states.to(self.torch_dtype)
        
#         with torch.no_grad():
#             self.model.eval()
#             print(f"[compute_estimated_reward] Rank {self.rank}: Model in eval mode")
#             estimated_rewards = self.model(hidden_states)  # (batch_size,)
        
#         # if not self.use_analytical:
#         #     reward_min, reward_max = estimated_rewards.min().item(), estimated_rewards.max().item()
#         #     print(f"[compute_estimated_reward] Rank {self.rank}: Output range: [{reward_min:.4f}, {reward_max:.4f}]")
#         #     if not (0 <= reward_min <= 1 and 0 <= reward_max <= 1):
#         #         print(f"[WARNING] Rank {self.rank}: Output range [{reward_min:.4f}, {reward_max:.4f}] outside [0,1]!")
        
#         print(f"[compute_estimated_reward] Rank {self.rank}: Output estimated_rewards shape: {estimated_rewards.shape}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Estimated rewards (first 5): {estimated_rewards[:5] if estimated_rewards.numel() > 0 else 'N/A'}")
#         print(f"[compute_estimated_reward] Rank {self.rank}: Mean: {estimated_rewards.mean().item():.6f}, Std: {estimated_rewards.std().item():.6f}")
        
#         output = DataProto.from_dict(tensors={"estimated_rewards": estimated_rewards})
#         output = output.to("cpu")
#         print(f"[compute_estimated_reward] Rank {self.rank}: === Reward estimation completed ===")
#         return output
    
#     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_estimator"))
#     def update_estimator(self, data: DataProto) -> DataProto:
#         """更新估计器 (重构版)"""
#         data = data.to(get_device_id())
#         hidden_states = data.batch["hidden_states"].to(self.torch_dtype)
#         target_rewards = data.batch["token_level_rewards"]
        
#         if len(target_rewards.shape) == 2:
#             target_rewards = target_rewards.sum(dim=-1)
        
#         target_rewards = target_rewards.to(self.torch_dtype)

#         if self.use_analytical:
#             # --- 解析解分支 (保持不变) ---
#             # (您的原始解析解代码放在这里)
#             # ...
#             # 为了简洁，此处省略，请使用您原来的代码
#             pass # Placeholder for your analytical code
#         else:
#             # --- 神经网络训练分支 ---
#             print(f"[update_estimator] Rank {self.rank}: --- Neural Net Update (Arch: {self.estimator_arch}) ---")
#             self.model.train()
            
#             # 前向传播
#             estimated_rewards = self.model(hidden_states)
            
#             # 计算损失
#             # 特别注意：对于BCE，目标值必须在[0,1]之间
#             if self.estimator_arch == 'mlp_sigmoid_bce':
#                 # 确保 target 在 [0, 1] 范围内，否则 BCELoss 会报错
#                 target_rewards.clamp_(0.0, 1.0)
            
#             loss = self.loss_fn(estimated_rewards, target_rewards)
            
#             # 反向传播
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             # 梯度裁剪
#             grad_norm = torch.nn.utils.clip_grad_norm_(
#                 self.model.parameters(), max_norm=self.gradient_clip_norm
#             )
            
#             self.optimizer.step()
            
#             # 更新学习率
#             if self.lr_scheduler:
#                 self.lr_scheduler.step()
            
#             current_lr = self.optimizer.param_groups[0]['lr']
            
#             metrics = {
#                 "reward_estimator/loss": loss.item(),
#                 "reward_estimator/mean_estimated": estimated_rewards.mean().item(),
#                 "reward_estimator/mean_target": target_rewards.mean().item(),
#                 "reward_estimator/grad_norm": grad_norm.item(),
#                 "reward_estimator/learning_rate": current_lr,
#             }
            
#             print(f"[update_estimator] Rank {self.rank}: Loss: {loss.item():.6f} | Grad norm: {grad_norm.item():.4f} | LR: {current_lr:.2e}")
#             print(f"[update_estimator] Rank {self.rank}: Estimated range: [{estimated_rewards.min().item():.4f}, {estimated_rewards.max().item():.4f}]")
#             print(f"[update_estimator] Rank {self.rank}: Target range:    [{target_rewards.min().item():.4f}, {target_rewards.max().item():.4f}]")

#         output = DataProto(meta_info={"metrics": metrics})
#         output = output.to('cpu')
#         print(f"[update_estimator] Rank {self.rank}: === Update completed ===")
#         return output
    
#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def save_checkpoint(self, path):
#         """保存检查点"""
#         print(f"[save_checkpoint] Rank {self.rank}: Saving checkpoint to {path}")
#         checkpoint = {
#             'model_state_dict': self.model.state_dict(),
#             'use_analytical': self.use_analytical,
#             'torch_dtype': self.torch_dtype,
#         }
        
#         if self.use_analytical:
#             # 保存累积统计量
#             print(f"[save_checkpoint] Rank {self.rank}: Saving analytical mode stats (n_samples={self.n_samples})")
#             checkpoint['XTX'] = self.XTX
#             checkpoint['XTy'] = self.XTy
#             checkpoint['n_samples'] = self.n_samples
#         else:
#             # 保存优化器状态
#             print(f"[save_checkpoint] Rank {self.rank}: Saving SGD mode optimizer state")
#             checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
#             # 保存神经网络配置
#             checkpoint['hidden_layer_size'] = self.hidden_layer_size
#             checkpoint['dropout_rate'] = self.dropout_rate
#             checkpoint['learning_rate'] = self.learning_rate
        
#         torch.save(checkpoint, path)
#         print(f"[save_checkpoint] Rank {self.rank}: Checkpoint saved successfully")
        
#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def load_checkpoint(self, path):
#         """加载检查点"""
#         if path is None:
#             print(f"[load_checkpoint] Rank {self.rank}: Path is None, skipping checkpoint load")
#             return
            
#         print(f"[load_checkpoint] Rank {self.rank}: Loading checkpoint from {path}")
#         checkpoint = torch.load(path, map_location=get_device_id())
        
#         # 检查模式是否匹配
#         if checkpoint['use_analytical'] != self.use_analytical:
#             print(f"[WARNING] Rank {self.rank}: Checkpoint mode {checkpoint['use_analytical']} != current mode {self.use_analytical}")
#             print(f"[load_checkpoint] Rank {self.rank}: Attempting to load anyway...")
        
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"[load_checkpoint] Rank {self.rank}: Model state dict loaded")
        
#         # 检查 dtype
#         if 'torch_dtype' in checkpoint:
#             loaded_dtype = checkpoint['torch_dtype']
#             print(f"[load_checkpoint] Rank {self.rank}: Checkpoint dtype: {loaded_dtype}")
#             if loaded_dtype != self.torch_dtype:
#                 print(f"[WARNING] Rank {self.rank}: Checkpoint dtype {loaded_dtype} != current config dtype {self.torch_dtype}")
        
#         if self.use_analytical and 'XTX' in checkpoint:
#             # 加载累积统计量
#             self.XTX = checkpoint['XTX'].to(get_device_id())
#             self.XTy = checkpoint['XTy'].to(get_device_id())
#             self.n_samples = checkpoint['n_samples']
#             print(f"[load_checkpoint] Rank {self.rank}: Loaded analytical estimator with {self.n_samples} accumulated samples")
#         elif not self.use_analytical and 'optimizer_state_dict' in checkpoint:
#             # 加载优化器状态
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             print(f"[load_checkpoint] Rank {self.rank}: Loaded SGD estimator with optimizer state")
            
#             # 加载神经网络配置（如果存在）
#             if 'hidden_layer_size' in checkpoint:
#                 print(f"[load_checkpoint] Rank {self.rank}: Checkpoint hidden_layer_size: {checkpoint['hidden_layer_size']}")
#             if 'dropout_rate' in checkpoint:
#                 print(f"[load_checkpoint] Rank {self.rank}: Checkpoint dropout_rate: {checkpoint['dropout_rate']}")
#         else:
#             print(f"[load_checkpoint] Rank {self.rank}: No appropriate state found for current mode")
#         print(f"[load_checkpoint] Rank {self.rank}: Checkpoint loading completed")
    
#     def reset_accumulation(self):
#         """重置累积的统计量（仅对解析解模式有效）"""
#         print(f"[reset_accumulation] Rank {self.rank}: Reset request received")
#         if self.use_analytical:
#             hidden_size = self.XTX.shape[0]
#             self.XTX = torch.zeros(hidden_size, hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.XTy = torch.zeros(hidden_size, device=get_device_id(), dtype=self.torch_dtype)
#             self.n_samples = 0
#             print(f"[reset_accumulation] Rank {self.rank}: Reset accumulated statistics for analytical estimator")
#         else:
#             print(f"[reset_accumulation] Rank {self.rank}: Warning: reset_accumulation only works for analytical mode (current mode: Neural Net SGD)")
class RunningMeanStd:
    """
    一个辅助类，用于在线（online）计算运行中的均值和方差。
    """
    def __init__(self, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']

    def to(self, device):
        """将内部张量移动到指定设备。"""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self


class RewardEstimatorWorker(Worker, DistProfilerExtension):
    """
    奖励估计器 Worker，增加了手动CPU Offload功能以节省显存。
    """
    def __init__(self, config: FSDPCriticConfig):
        # -- 基础初始化 --
        Worker.__init__(self)
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.log_adapter = logging.LoggerAdapter(self.logger, {'rank': self.rank})
        
        self._init_profiler(config)
        self._init_distributed(config)

        self.config = config
        
        # -- 核心参数 --
        self.hidden_size = self.config.model.get("hidden_size", 3584)
        torch_dtype_str = self.config.model.fsdp_config.get("model_dtype", "fp32")
        from verl.utils.torch_dtypes import PrecisionType
        self.torch_dtype = PrecisionType.to_dtype(torch_dtype_str)

        # -- 模式选择 --
        self.mode = self.config.get("mode", "sgd") 
        
        self.lambda_reg = self.config.get("lambda_reg", 1e-3)
        self.learning_rate = self.config.get("learning_rate", 1e-3)

        # -- 归一化与平滑 --
        self.normalize_features = self.config.get("normalize_features", False)
        self.normalize_value = self.config.get("normalize_value", False)
        self.use_target_ema = self.config.get("use_target_ema", False)
        self.ema_alpha = self.config.get("ema_alpha", 0.7)

        # -- 新增：Offload 配置 --
        self.offload_to_cpu = self.config.get("offload_to_cpu", True)
        print(f"CPU Offload is {'ENABLED' if self.offload_to_cpu else 'DISABLED'}.")
        
        print(f"Worker 初始化完成。模式: '{self.mode}', 目标数据类型: {self.torch_dtype}")
        
    def _init_profiler(self, config: FSDPCriticConfig):
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        tool = omega_profiler_config.get("tool")
        tool_config = omega_conf_to_dataclass(
            omega_profiler_config.get("tool_config", {}).get(tool)
        ) if tool in ["npu", "nsys", "torch", "torch_memory"] else None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    def _init_distributed(self, config: FSDPCriticConfig):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        world_size = torch.distributed.get_world_size()
        
        fsdp_size = config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        
        self.ulysses_sequence_parallel_size = config.get("ulysses_sequence_parallel_size", 1)
        if self.ulysses_sequence_parallel_size > 1:
            dp_size = world_size // self.ulysses_sequence_parallel_size
            self.ulysses_device_mesh = torch.distributed.device_mesh.init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            dp_rank = self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            self.ulysses_device_mesh = None
            is_collect = True
            dp_rank = self.rank
            
        self._register_dispatch_collect_info("reward_estimator", dp_rank=dp_rank, is_collect=is_collect)

    # --- 新增：手动 Offload 和 Load 的辅助函数 ---
    def _load_to_gpu(self):
        """将所有模型组件和状态加载到 GPU。"""
        if not self.offload_to_cpu:
            return
        
        device = get_device_id()
        self.log_adapter.debug("Loading RewardEstimator to GPU...")
        self.model.to(device)
        if self.normalize_features: self.feature_rms.to(device)
        if self.normalize_value: self.value_rms.to(device)
        
        if self.mode == 'analytical':
            self.XTX = self.XTX.to(device)
            self.XTy = self.XTy.to(device)
        elif self.mode == 'rls':
            self.P = self.P.to(device)
        elif self.mode == 'sgd':
            # 优化器状态需要特殊处理
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        self.log_adapter.debug("RewardEstimator loaded to GPU.")

    def _offload_to_cpu(self):
        """将所有模型组件和状态卸载到 CPU。"""
        if not self.offload_to_cpu:
            return
            
        self.log_adapter.debug("Offloading RewardEstimator to CPU...")
        self.model.to('cpu')
        if self.normalize_features: self.feature_rms.to('cpu')
        if self.normalize_value: self.value_rms.to('cpu')
        
        if self.mode == 'analytical':
            self.XTX = self.XTX.to('cpu')
            self.XTy = self.XTy.to('cpu')
        elif self.mode == 'rls':
            self.P = self.P.to('cpu')
        elif self.mode == 'sgd':
            # 优化器状态也需要移到CPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cpu()
        
        aggressive_empty_cache() # 清理显存碎片
        self.log_adapter.debug("RewardEstimator offloaded to CPU.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """初始化模型。如果启用offload，则在CPU上初始化以节省显存。"""
        # 决定初始化的设备
        device = 'cpu' if self.offload_to_cpu else get_device_id()
        print(f"在设备 {device} 上初始化模型...")
        
        self.model = torch.nn.Linear(self.hidden_size, 1, bias=False).to(device, dtype=self.torch_dtype)
        self.n_samples = 0
        
        if self.normalize_features:
            self.feature_rms = RunningMeanStd(shape=(self.hidden_size,), device=device)
        if self.normalize_value:
            self.value_rms = RunningMeanStd(shape=(), device=device)
        if self.use_target_ema:
            self.ema_target_value = None

        if self.mode == 'analytical':
            self.XTX = torch.zeros(self.hidden_size, self.hidden_size, device=device, dtype=self.torch_dtype)
            self.XTy = torch.zeros(self.hidden_size, device=device, dtype=self.torch_dtype)
            print("模式 'analytical' 已初始化。")
        elif self.mode == 'rls':
            self.P = torch.eye(self.hidden_size, device=device, dtype=self.torch_dtype) / self.lambda_reg
            print(f"模式 'rls' 已初始化。")
        elif self.mode == 'sgd':
            # 优化器需要模型参数，此时模型可能在CPU上
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            print(f"模式 'sgd' 已初始化。")
        else:
            raise ValueError(f"未知的模式: {self.mode}。")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_estimator"))
    def compute_estimated_reward(self, data: DataProto) -> DataProto:
        """计算估计的奖励基线，并在计算前后自动处理GPU加载/卸载。"""
        try:
            self._load_to_gpu() # 将模型加载到GPU
            
            data = data.to(get_device_id())
            hidden_states = data.batch["hidden_states"].to(self.torch_dtype)
            
            self.model.eval()
            with torch.no_grad():
                if self.normalize_features:
                    hidden_states = (hidden_states - self.feature_rms.mean) / torch.sqrt(self.feature_rms.var + 1e-8)

                estimated_rewards = self.model(hidden_states).squeeze(-1)
                
                if self.normalize_value:
                    estimated_rewards = estimated_rewards * torch.sqrt(self.value_rms.var + 1e-8) + self.value_rms.mean

            output = DataProto.from_dict(tensors={"estimated_rewards": estimated_rewards.cpu()})
            return output
        finally:
            self._offload_to_cpu() # 确保计算结束后将模型卸载回CPU
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_estimator"))
    def update_estimator(self, data: DataProto) -> DataProto:
        """更新估计器，并在计算前后自动处理GPU加载/卸载。"""
        try:
            self._load_to_gpu() # 将模型和状态加载到GPU

            # --- 阶段 1: 数据准备 ---
            data = data.to(get_device_id())
            # (后续代码与上一版相同，此处省略以保持简洁)
            hidden_states = data.batch["hidden_states"].to(self.torch_dtype)
            target_rewards = data.batch["token_level_rewards"]
            if target_rewards.dim() == 2: target_rewards = target_rewards.sum(dim=-1)
            
            # --- 阶段 2 & 3: 平滑与归一化 ---
            if self.use_target_ema:
                current_mean_reward = target_rewards.mean()
                if self.ema_target_value is None: self.ema_target_value = current_mean_reward.cpu()
                self.ema_target_value = self.ema_alpha * self.ema_target_value.to(current_mean_reward.device) + (1 - self.ema_alpha) * current_mean_reward
                final_target = target_rewards - current_mean_reward + self.ema_target_value.detach()
            else:
                final_target = target_rewards
            if self.normalize_value: self.value_rms.update(final_target.unsqueeze(-1))
            if self.normalize_features: self.feature_rms.update(hidden_states)
            if self.normalize_features: norm_hidden_states = (hidden_states - self.feature_rms.mean) / torch.sqrt(self.feature_rms.var + 1e-8)
            else: norm_hidden_states = hidden_states
            if self.normalize_value: norm_target = (final_target - self.value_rms.mean) / torch.sqrt(self.value_rms.var + 1e-8)
            else: norm_target = final_target
            
            # --- 阶段 4: 更新逻辑 ---
            metrics = {}
            self.n_samples += norm_hidden_states.shape[0]
            if self.mode == 'analytical':
                self.XTX += norm_hidden_states.T @ norm_hidden_states
                self.XTy += norm_hidden_states.T @ norm_target
                try:
                    XTX_reg = self.XTX + self.lambda_reg * torch.eye(self.XTX.shape[0], device=get_device_id(), dtype=self.torch_dtype)
                    theta = torch.linalg.solve(XTX_reg, self.XTy)
                    with torch.no_grad(): self.model.weight.copy_(theta.unsqueeze(0))
                    metrics["reward_estimator/XTX_cond"] = torch.linalg.cond(XTX_reg).item()
                except torch.linalg.LinAlgError as e:
                    print(f"解析解模式下发生线性代数错误: {e}")
            elif self.mode == 'rls':
                with torch.no_grad():
                    for i in range(norm_hidden_states.shape[0]):
                        x = norm_hidden_states[i]
                        y = norm_target[i]
                        Px = self.P @ x
                        k_denominator = 1.0 + x @ Px
                        K = Px / k_denominator
                        current_theta = self.model.weight.squeeze(0)
                        prediction_error = y - (current_theta @ x)
                        new_theta = current_theta + K * prediction_error
                        self.model.weight.copy_(new_theta.unsqueeze(0))
                        self.P -= torch.outer(K, Px)
            elif self.mode == 'sgd':
                self.model.train()
                estimated_rewards = self.model(norm_hidden_states).squeeze(-1)
                loss = torch.nn.functional.mse_loss(estimated_rewards, norm_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metrics["reward_estimator/loss"] = loss.item()

            # --- 阶段 5: 计算指标 ---
            with torch.no_grad():
                estimated_rewards_norm = self.model(norm_hidden_states).squeeze(-1)
                loss = torch.nn.functional.mse_loss(estimated_rewards_norm, norm_target)
                if 'reward_estimator/loss' not in metrics:
                    metrics['reward_estimator/loss'] = loss.item()
            mean_estimated_denorm = estimated_rewards_norm.mean() * torch.sqrt(self.value_rms.var + 1e-8) + self.value_rms.mean if self.normalize_value else estimated_rewards_norm.mean()
            metrics.update({
                "reward_estimator/n_samples": self.n_samples,
                "reward_estimator/mean_estimated": mean_estimated_denorm.item(),
                "reward_estimator/mean_target": target_rewards.mean().item(),
                "reward_estimator/ema_target": self.ema_target_value.item() if self.use_target_ema and self.ema_target_value is not None else -1,
            })
            
            return DataProto(meta_info={"metrics": metrics}).to('cpu')
        
        finally:
            self._offload_to_cpu() # 确保计算结束后将模型和状态卸载回CPU

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path):
        # 保存时，所有东西都应该在CPU上（如果offload开启），所以可以直接保存
        print(f"正在保存检查点到 {path} (确保组件在CPU上)...")
        self._offload_to_cpu() # 确保所有组件都在CPU上以便统一保存
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'mode': self.mode,
            'torch_dtype': self.torch_dtype,
            'feature_rms_state_dict': self.feature_rms.state_dict() if self.normalize_features else None,
            'value_rms_state_dict': self.value_rms.state_dict() if self.normalize_value else None,
            'ema_target_value': self.ema_target_value if self.use_target_ema else None,
        }
        
        if self.mode == 'analytical':
            checkpoint.update({'XTX': self.XTX, 'XTy': self.XTy, 'n_samples': self.n_samples})
        elif self.mode == 'rls':
            checkpoint.update({'P': self.P, 'n_samples': self.n_samples})
        elif self.mode == 'sgd':
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print("检查点保存成功。")
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path):
        if not path or not os.path.exists(path):
            self.log_adapter.warning(f"检查点路径 '{path}' 无效或不存在。跳过加载。")
            return
            
        # 根据是否offload，决定加载到哪个设备
        device = 'cpu' if self.offload_to_cpu else get_device_id()
        print(f"正在从 {path} 加载检查点到设备 {device}...")
        checkpoint = torch.load(path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.normalize_features and checkpoint.get('feature_rms_state_dict'):
            self.feature_rms.load_state_dict(checkpoint['feature_rms_state_dict'])
        if self.normalize_value and checkpoint.get('value_rms_state_dict'):
            self.value_rms.load_state_dict(checkpoint['value_rms_state_dict'])
        if self.use_target_ema and checkpoint.get('ema_target_value') is not None:
            self.ema_target_value = checkpoint['ema_target_value']

        mode_in_ckpt = checkpoint.get('mode', 'sgd') 
        if mode_in_ckpt != self.mode:
            self.log_adapter.warning(f"检查点模式 '{mode_in_ckpt}' 与当前配置模式 '{self.mode}' 不匹配。只加载模型权重。")
            return

        if self.mode == 'analytical' and 'XTX' in checkpoint:
            self.XTX = checkpoint['XTX']; self.XTy = checkpoint['XTy']; self.n_samples = checkpoint['n_samples']
        elif self.mode == 'rls' and 'P' in checkpoint:
            self.P = checkpoint['P']; self.n_samples = checkpoint['n_samples']
        elif self.mode == 'sgd' and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 确保加载后所有组件都在正确的设备上
        if self.offload_to_cpu:
            self._offload_to_cpu()
        else:
            self._load_to_gpu()
            
        print("检查点加载完成。")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_accumulation(self):
        """重置累积统计信息，并确保在正确的设备上。"""
        print("收到重置累积统计的请求。")
        device = get_device_id() if not self.offload_to_cpu else 'cpu'
        self.n_samples = 0

        if self.mode == 'analytical':
            self.XTX = torch.zeros(self.hidden_size, self.hidden_size, device=device, dtype=self.torch_dtype)
            self.XTy = torch.zeros(self.hidden_size, device=device, dtype=self.torch_dtype)
            print("已重置 'analytical' 模式的累积统计信息。")
        elif self.mode == 'rls':
            self.P = torch.eye(self.hidden_size, device=device, dtype=self.torch_dtype) / self.lambda_reg
            print("已重置 'rls' 模式的累积统计信息。")
        else:
            self.log_adapter.warning(f"重置操作对 '{self.mode}' 模式无效。")

# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
