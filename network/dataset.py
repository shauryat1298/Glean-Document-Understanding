import torch
from torch.utils import data
import os
from pathlib import Path
from src.Glean.utils import Neighbour, candidate, annotation_parser, operations as op, preprocess
from src.Glean.entity.config_entity import TrainModelConfig
from src.Glean import logger
import pickle

class DocumentsDataset(data.Dataset):
    """Stores the annotated documents dataset."""
    
    def __init__(self, config:TrainModelConfig, split_name='train'):
        """ Initialize the dataset with preprocessing """
        self.config = config
        os.makedirs(self.config.cached_data_dir, exist_ok=True)
        cached_data_path = Path(self.config.cached_data_dir) / f"cached_data_{split_name}.pickle"
        if cached_data_path.exists():
            logger.info("Preprocessed data available, Loading data from cache...")
            with open(cached_data_path, "rb") as f:
                cached_data = pickle.load(f)
            classes_count = cached_data['count']
            class_mapping = cached_data['mapping']
            print("\nClass Mapping:", class_mapping)
            print("Classs counts:", classes_count)
            _data = cached_data['data']
            self.vocab = cached_data['vocab']
            self.field_ids, self.candidate_cords, self.neighbours, self.neighbour_cords, self.mask, self.labels = _data
        else:
            logger.info("Preprocessed data not available")
            annotation, classes_count, class_mapping = annotation_parser.get_data(self.config, split_name)
            print("Class Mapping:", class_mapping)
            print("Classs counts:", classes_count)
            annotation = candidate.attach_candidate(annotation, config.candidate_dir)
            annotation, self.vocab = Neighbour.attach_neighbour(annotation, config.ocr_dir, vocab_size=config.vocab_size)
            annotation = op.normalize_positions(annotation)
            _data = preprocess.parse_input(annotation, class_mapping, config.neighbours, self.vocab)
            self.field_ids, self.candidate_cords, self.neighbours, self.neighbour_cords, self.mask, self.labels = _data
            cached_data = {'count': classes_count, "mapping": class_mapping, 'vocab': self.vocab, 'data': _data}
            logger.info("Saving Cache..")
            with open(cached_data_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Done !!")
    
    def __len__(self):
        return len(self.field_ids)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.field_ids[idx]).type(torch.FloatTensor),
            torch.tensor(self.candidate_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.neighbours[idx]),
            torch.tensor(self.neighbour_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.mask[idx]).type(torch.FloatTensor),
            torch.tensor(self.labels[idx]).type(torch.FloatTensor)
        )
