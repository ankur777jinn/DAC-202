"""
Central configuration -- A100-optimized.

Key changes from the 2-GPU baseline:
  - Single-GPU friendly (no torchrun needed; just `python train_stage2.py ...`)
  - Larger batch sizes that fit comfortably in A100 40GB/80GB
  - LR scaled to the larger batch
  - More dataloader workers (A100 is fast; CPU was the bottleneck)
  - bf16 instead of fp16 (A100 tensor cores prefer it)

Set the LITS_DATA_ROOT env var to point to your data directory, e.g.:
  export LITS_DATA_ROOT=/path/to/lits_data
or on Windows:
  set LITS_DATA_ROOT=C:\\path\\to\\lits_data
"""
import os
from dataclasses import dataclass

_DATA_ROOT = os.environ.get("LITS_DATA_ROOT", "./lits_data")


@dataclass
class DataConfig:
    csv_path: str = os.path.join(_DATA_ROOT, "lits_train.csv")
    train_csv: str = os.path.join(_DATA_ROOT, "lits_train.csv")
    test_csv: str = os.path.join(_DATA_ROOT, "lits_test.csv")
    probe_csv: str = os.path.join(_DATA_ROOT, "lits_probe.csv")
    data_root: str = os.path.join(_DATA_ROOT, "dataset_6", "dataset_6")
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
    # ---------------- Stage 1 ----------------
    stage1_epochs: int = 30          # already converges fast; 30 is enough
    stage1_batch_size: int = 64      # A100 has tons of memory
    stage1_lr: float = 2e-3          # scaled with batch (vs 16->1e-3)

    # ---------------- Stage 2 ----------------
    stage2_epochs: int = 20           # was 60 — 25 will give converged-enough Dice
    stage2_batch_size: int = 128      # A100 can handle this for ResNet-34
    stage2_lr: float = 1e-3          # scaled with batch (vs 24->5e-4)

    weight_decay: float = 1e-5
    num_workers: int = 4             # safe default; bump to 12 on beefy machines
    prefetch_factor: int = 4         # each worker preloads this many batches
    use_amp: bool = True
    use_bf16: bool = True            # A100 tensor cores prefer bf16
    grad_clip: float = 1.0

    warmup_epochs: int = 2
    min_lr: float = 1e-6

    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./runs"
    save_every: int = 10
    seed: int = 42

    # ---------------- A100 perf knobs ----------------
    compile_model: bool = False      # torch.compile() — disabled by default (issues on Windows)
    channels_last: bool = True       # NHWC memory format = faster on A100


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