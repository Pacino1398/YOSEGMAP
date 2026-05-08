from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.paths import (
    get_data_yaml,
    get_default_mask_dir,
    get_repo_root,
    get_runs_dir,
    get_test_input_dir,
    get_yolo_dir,
    find_default_weights,
)


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path
    yolo_root: Path
    data_yaml: Path
    runs_dir: Path
    default_source: Path
    default_weights: Path
    default_run_name: str = "exp"
    default_device: str = "0"
    default_conf_thres: float = 0.25
    default_grid_scale: int = 10

    @property
    def default_mask_dir(self) -> Path:
        return get_default_mask_dir(self.default_run_name)


DEFAULT_CONFIG = ProjectConfig(
    repo_root=get_repo_root(),
    yolo_root=get_yolo_dir(),
    data_yaml=get_data_yaml(),
    runs_dir=get_runs_dir(),
    default_source=get_test_input_dir(),
    default_weights=find_default_weights(),
)
