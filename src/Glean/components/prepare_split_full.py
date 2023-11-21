import gzip
import json
from pathlib import Path
from random import shuffle
import os
from src.Glean import logger
from src.Glean.entity.config_entity import PrepareSplitConfig

class PrepareSplit:
    def __init__(self, config:PrepareSplitConfig):
        self.config = config
        self.train_test_split_ratio = 0.8
    
    def split_file(self):
        try:
            train_test_split_path = Path(self.config.source_dir_full)
            save_dir = self.config.save_dir

            pdf_files = [x.stem for x in train_test_split_path.glob("*.pdf")]
            shuffle(pdf_files)
            split_point = int(len(pdf_files)*self.train_test_split_ratio)

            train_set = pdf_files[:split_point]
            val_set = pdf_files[split_point:]

            with open(os.path.join(save_dir, 'train.txt'), 'w') as train_file:
                train_file.write("\n".join(train_set))
            logger.info(f"Saved training split into {save_dir}/train.txt")

            # Save "valid" data to val.txt
            with open(os.path.join(save_dir, 'val.txt'), 'w') as valid_file:
                valid_file.write("\n".join(val_set))
            logger.info(f"Saved validation split into {save_dir}/val.txt")
        
        except Exception as e:
            raise e

