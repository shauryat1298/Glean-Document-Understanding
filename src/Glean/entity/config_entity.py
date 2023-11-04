from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PrepareSplitConfig:
    root_dir: Path
    source_dir: Path
    save_dir: Path

@dataclass(frozen=True)
class GenerateOCRResultsConfig:
    root_dir: Path
    source_dir: Path
    save_dir: Path

@dataclass(frozen=True)
class GroundTruthAnnotationsConfig:
    root_dir: Path
    source_dir: Path
    save_dir: Path

@dataclass(frozen=True)
class ExtractCandidatesConfig:
    root_dir: Path
    ocr_dir: Path
    candidates_dir: Path

@dataclass(frozen=True)
class TrainModelConfig:
    root_dir: Path
    cached_data_dir: Path
    ground_truth_dir: Path
    candidate_dir: Path
    ocr_dir: Path
    split_dir: Path
    best_model_dir: Path
    neighbours: int
    heads: int
    embedding_size: int
    vocab_size: int
    batch_size: int
    epochs: int
    lr: float