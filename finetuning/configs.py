"""Configuration presets for ARBITER fine-tuning and adaptation.

Provides centralized configurations for:
1. "typed": Mono-task specialist on Typed Decisions (76.45% accuracy).
2. "mixture_80k": Balanced 80,000 decisions mixture boosted in typed reasoning (32k typed, 24k policy, 24k NLI),
   without historical benchmark replay (Banking77 / MASSIVE removed).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AdaptationConfig:
    name: str
    description: str
    run_name: str
    data_dir: Path
    train_path: Path
    validation_path: Path
    output_dir: Path
    epochs: int
    batch_size: int = 4
    accumulation: int = 8
    max_length: int = 1024
    backbone_lr: float = 1e-5
    head_lr: float = 4e-5
    clip: float = 3.0
    eval_every: int = 50
    seed: int = 13
    deterministic: bool = True
    heldout: List[Path] = field(default_factory=lambda: [
        Path("benchmark/data/cases.jsonl"),
        Path("benchmark/data/public_requested_v1/cases.jsonl"),
    ])

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "run_name": self.run_name,
            "data_dir": str(self.data_dir),
            "train_path": str(self.train_path),
            "validation_path": str(self.validation_path),
            "output_dir": str(self.output_dir),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accumulation": self.accumulation,
            "max_length": self.max_length,
            "backbone_lr": self.backbone_lr,
            "head_lr": self.head_lr,
            "clip": self.clip,
            "eval_every": self.eval_every,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }


def get_config(name: str, root_dir: Optional[Path] = None) -> AdaptationConfig:
    """Retrieve an adaptation configuration by name."""
    root = Path(root_dir) if root_dir else Path.cwd()
    
    if name == "typed":
        data_dir = root / "results/adaptation/typed"
        return AdaptationConfig(
            name="typed",
            description="Spécialiste Typed Decisions mono-tâche (atteint 76.45% sur le benchmark officiel)",
            run_name="arbiter-targeted-typed-v1",
            data_dir=data_dir,
            train_path=data_dir / "train.jsonl",
            validation_path=data_dir / "validation.jsonl",
            output_dir=root / "results/arbiter-targeted-typed-v1",
            epochs=3,
            backbone_lr=1e-5,
            head_lr=4e-5,
            eval_every=50,
            heldout=[
                root / "benchmark/data/cases.jsonl",
                root / "benchmark/data/public_requested_v1/cases.jsonl",
            ],
        )
    elif name in ("mixture_80k", "mixture-80k", "80k"):
        data_dir = root / "results/adaptation/80k_typed"
        return AdaptationConfig(
            name="mixture_80k",
            description="Mélange 80,000 décisions enrichi en typé (32k typé, 24k politique, 24k NLI) sans replay historique",
            run_name="arbiter-targeted-80k-v2",
            data_dir=data_dir,
            train_path=data_dir / "train.jsonl",
            validation_path=data_dir / "validation.jsonl",
            output_dir=root / "results/arbiter-targeted-80k-v2",
            epochs=1,
            backbone_lr=1e-5,
            head_lr=4e-5,
            eval_every=250,
            heldout=[
                root / "benchmark/data/cases.jsonl",
                root / "benchmark/data/public_requested_v1/cases.jsonl",
            ],
        )
    else:
        raise ValueError(f"Unknown config '{name}'. Available configs: 'typed', 'mixture_80k'")


AVAILABLE_CONFIGS = ["typed", "mixture_80k"]
