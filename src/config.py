import yaml
from pydantic import BaseModel
from pathlib import Path

class DatasetConfig(BaseModel):
    name: str
    version: str
    sample_rate: int
    clip_duration_seconds: float

class PathsConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    results_dir: str

class BenchmarkConfig(BaseModel):
    seed: int
    dataset: DatasetConfig
    paths: PathsConfig

def load_config(config_path: str | Path = "configs/base.yaml") -> BenchmarkConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig(**data)
