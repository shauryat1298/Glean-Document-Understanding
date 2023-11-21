from src.Glean.constants import *
from src.Glean.utils.common import read_yaml, create_directories
from src.Glean.entity.config_entity import PrepareSplitConfig, GenerateOCRResultsConfig, GroundTruthAnnotationsConfig, ExtractCandidatesConfig, TrainModelConfig, EvaluationConfig
import os

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def prepare_split_config(self) -> PrepareSplitConfig:
        config = self.config.prepare_split

        create_directories([config.root_dir])

        prepare_split_config = PrepareSplitConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            source_dir_full=config.source_dir_full,
            save_dir=config.save_dir
        )

        return prepare_split_config
    
    def generate_ocr_results_config(self) -> GenerateOCRResultsConfig:
        config = self.config.generate_ocr_results

        create_directories([config.root_dir])

        generate_ocr_results_config = GenerateOCRResultsConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            save_dir=config.save_dir
        )

        return generate_ocr_results_config
    
    def ground_truth_annotations_config(self) -> GroundTruthAnnotationsConfig:
        config = self.config.ground_truth_annotations

        create_directories([config.root_dir])

        ground_truth_annotations_config = GroundTruthAnnotationsConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            save_dir=config.save_dir
        )

        return ground_truth_annotations_config
    
    def extract_candidates_config(self) -> ExtractCandidatesConfig:
        config = self.config.extract_candidates

        create_directories([config.root_dir])

        extract_candidates_config = ExtractCandidatesConfig(
            root_dir=config.root_dir,
            ocr_dir=config.ocr_dir,
            candidates_dir=config.candidates_dir
        )

        return extract_candidates_config
    
    def train_model_config(self) -> TrainModelConfig:
        config = self.config.train_model

        create_directories([config.root_dir])

        train_model_config = TrainModelConfig(
            root_dir=config.root_dir,
            cached_data_dir=config.cached_data_dir,
            ground_truth_dir=config.ground_truth_dir,
            candidate_dir=config.candidates_dir,
            ocr_dir=config.ocr_dir,
            split_dir=config.split_dir,
            best_model_dir=config.best_model,
            neighbours=self.params.NEIGHBOURS,
            heads=self.params.HEADS,
            embedding_size=self.params.EMBEDDING_SIZE,
            vocab_size=self.params.VOCAB_SIZE,
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.EPOCHS,
            lr=self.params.LR,
            mlflow_uri="https://dagshub.com/shauryat1298/Glean-Document-Understanding.mlflow",
            all_params=self.params
        )

        return train_model_config
    
    def evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluate_model

        create_directories([config.root_dir])

        evalutate_config = EvaluationConfig(
            root_dir=config.root_dir,
            best_model_dir=config.best_model,
            all_params=self.params,
            mlflow_uri="https://dagshub.com/shauryat1298/Glean-Document-Understanding.mlflow"
        )
        return evalutate_config
