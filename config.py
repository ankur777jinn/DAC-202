"""
Central configuration for the LiTS cascaded segmentation project.
All hyperparameters live here.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    # The CSV file lits_df.csv lists every slice with three columns:
    csv_path: str = "/teamspace/studios/this_studio/lits_data/lits_train.csv"
    train_csv: str = "/teamspace/studios/this_studio/lits_data/lits_train.csv"
    test_csv: str = "/teamspace/studios/this_studio/lits_data/lits_test.csv"
    probe_csv: str = "/teamspace/studios/this_studio/lits_data/lits_probe.csv"
    data_root: str = "/teamspace/studios/this_studio/lits_data/dataset_6/dataset_6"
    # If you have the pre-split CSVs already, point to them and we'll use them.
    # Otherwise we do our own patient-level split.
    use_provided_splits: bool = True          # used as validation

    # Fallback patient-level split ratios (used if use_provided_splits=False)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # Image processing
    image_size: int = 384
    lesion_crop_size: int = 256

    # Filtering rules (apply to lits_df rows)
    drop_empty_liver_for_stage1: bool = True   # ~33% of rows are liver-empty
    drop_empty_liver_for_stage2: bool = True   # no liver -> no tumor anyway
    drop_empty_tumor_for_stage2_train: bool = False  # keep tumor-empty slices as hard negatives during training
    drop_empty_tumor_for_stage2_eval: bool = False   # ALWAYS keep for eval (full test distribution)


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
    stage1_epochs: int = 40
    stage1_batch_size: int = 64
    stage1_lr: float = 1e-3

    stage2_epochs: int = 80
    stage2_batch_size: int = 24
    stage2_lr: float = 5e-4

    weight_decay: float = 1e-5
    num_workers: int = 8
    use_amp: bool = True
    grad_clip: float = 1.0

    warmup_epochs: int = 3
    min_lr: float = 1e-6

    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./runs"
    save_every: int = 5
    seed: int = 42


@dataclass
class ModelConfig:
    stage1_encoder: str = "resnet34"
    stage1_pretrained: bool = True

    stage2_encoder: str = "resnet34"
    stage2_pretrained: bool = True
    use_transformer: bool = True
    transformer_layers: int = 3
    transformer_heads: int = 8
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.1


@dataclass
class AblationConfig:
    name: str = "D_full"
    use_cascade: bool = True
    use_transformer: bool = True
    use_boundary_loss: bool = True
    use_focal_tversky: bool = True


ABLATIONS = {
    "A_baseline":              AblationConfig("A_baseline",              False, False, False, False),
    "B_cascade":               AblationConfig("B_cascade",               True,  False, False, False),
    "C_cascade_transformer":   AblationConfig("C_cascade_transformer",   True,  True,  False, False),
    "D_full":                  AblationConfig("D_full",                  True,  True,  True,  True),
}


DATA = DataConfig()
LOSS = LossConfig()
TRAIN = TrainConfig()
MODEL = ModelConfig()
