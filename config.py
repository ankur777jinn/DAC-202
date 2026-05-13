import os
from dataclasses import dataclass

# Auto-detect Lightning AI vs local
_LIGHTNING = '/teamspace/studios/this_studio/lits_data'
_DATA_ROOT = _LIGHTNING if os.path.isdir(_LIGHTNING) else os.environ.get('LITS_DATA_ROOT', './lits_data')

@dataclass
class DataConfig:
    csv_path: str = os.path.join(_DATA_ROOT, 'lits_train.csv')
    train_csv: str = os.path.join(_DATA_ROOT, 'lits_train.csv')
    test_csv: str = os.path.join(_DATA_ROOT, 'lits_test.csv')
    probe_csv: str = os.path.join(_DATA_ROOT, 'lits_probe.csv')
    data_root: str = os.path.join(_DATA_ROOT, 'dataset_6', 'dataset_6')
    use_provided_splits: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    image_size: int = 384
    lesion_crop_size: int = 192
    drop_empty_liver_for_stage1: bool = True
    drop_empty_liver_for_stage2: bool = True
    drop_empty_tumor_for_stage2_train: bool = False
    drop_empty_tumor_for_stage2_eval: bool = False

@dataclass
class LossConfig:
    alpha: float = 0.3
    beta: float = 0.7
    gamma: float = 1.3333
    boundary_k: float = 5.0
    lambda_focal_tversky: float = 1.0
    lambda_boundary: float = 0.5
    stage1_dice_weight: float = 1.0
    stage1_bce_weight: float = 1.0

@dataclass
class TrainConfig:
    stage1_epochs: int = 20
    stage1_batch_size: int = 64
    stage1_lr: float = 2e-3
    stage2_epochs: int = 20
    stage2_batch_size: int = 128
    stage2_lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 12
    prefetch_factor: int = 4
    use_amp: bool = True
    use_bf16: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    ckpt_dir: str = './checkpoints'
    log_dir: str = './runs'
    save_every: int = 10
    seed: int = 42
    compile_model: bool = False
    channels_last: bool = True

@dataclass
class ModelConfig:
    stage1_encoder: str = 'resnet34'
    stage1_pretrained: bool = True
    stage2_encoder: str = 'resnet34'
    stage2_pretrained: bool = True
    use_transformer: bool = True
    transformer_layers: int = 3
    transformer_heads: int = 8
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.1

@dataclass
class AblationConfig:
    name: str = 'D_full'
    use_cascade: bool = True
    use_transformer: bool = True
    use_boundary_loss: bool = True
    use_focal_tversky: bool = True

ABLATIONS = {
    'A_baseline':              AblationConfig('A_baseline',              False, False, False, False),
    'B_cascade':               AblationConfig('B_cascade',               True,  False, False, False),
    'C_cascade_transformer':   AblationConfig('C_cascade_transformer',   True,  True,  False, False),
    'D_full':                  AblationConfig('D_full',                  True,  True,  True,  True),
}

DATA = DataConfig()
LOSS = LossConfig()
TRAIN = TrainConfig()
MODEL = ModelConfig()
