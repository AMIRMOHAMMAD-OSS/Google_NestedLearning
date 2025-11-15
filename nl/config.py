from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CMSLevelCfg:
    d_ff: int
    update_every: int

@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int
    d_ff: int
    n_layers: int
    max_seq_len: int
    dropout: float
    d_kv: int
    cms_levels: List[CMSLevelCfg]
    inner_lr: float
    inner_scale_xtx: float = 1.0
    inner_apply_during_eval: bool = True
    inner_apply_during_sampling: bool = False

@dataclass
class TrainCfg:
    batch_size: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    max_steps: int
    warmup_steps: int
    eval_every: int
    ckpt_every: int
    log_every: int

@dataclass
class RootCfg:
    exp_name: str
    seed: int
    model: ModelCfg
    train: TrainCfg
    data: Dict[str, Any]
