import gzip
import json
from pathlib import Path
import os
from src.Glean import logger
from src.Glean.entity.config_entity import PrepareSplitConfig

class PrepareSplit:
    def __init__(self, config:PrepareSplitConfig):
        self.config = config
    
    def split_file(self):
        try:
            train_test_split_path = self.config.source_dir
            save_dir = self.config.save_dir

            with open(train_test_split_path) as f:
                data = json.load(f)
            
            train_data = data['train']
            with open(os.path.join(save_dir, 'train.txt'), 'w') as train_file:
                for entry in train_data:
                    file_name = os.path.splitext(entry)[0]
                    train_file.write(file_name + '\n')
            logger.info(f"Saved training split into {save_dir}/train.txt")

            # Save "valid" data to val.txt
            valid_data = data['valid']
            with open(os.path.join(save_dir, 'val.txt'), 'w') as valid_file:
                for entry in valid_data:
                    file_name = os.path.splitext(entry)[0]
                    valid_file.write(file_name + '\n')
            logger.info(f"Saved validation split into {save_dir}/val.txt")
        
        except Exception as e:
            raise e

