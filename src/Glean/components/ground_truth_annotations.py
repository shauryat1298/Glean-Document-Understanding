import gzip
import json
from pathlib import Path
import os
from tqdm import tqdm
from src.Glean import logger
from src.Glean.entity.config_entity import GroundTruthAnnotationsConfig

class GroundTruthAnnotations:
    def __init__(self, config:GroundTruthAnnotationsConfig):
        self.config = config
    
    def read_and_cut(self):
        try:
            file_path = self.config.source_dir
            output_dir = self.config.save_dir

            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    # if 'annotations' in data:
                    #     del data['annotations']
                    for page in data['ocr']['pages']:
                        if 'tokens' in page:
                            del page['tokens']
                        if 'lines' in page:
                            del page['lines']
                        if 'paragraphs' in page:
                            del page['paragraphs']
                        if 'blocks' in page:
                            del page['blocks']
                    filename = os.path.splitext(data['filename'])[0]
                    output_file_path = os.path.join(output_dir, filename+'.json')

                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        json.dump(data, output_file, ensure_ascii=False, indent=4)
                        
            logger.info(f"Ground Truth Annotations saved to {output_dir}")
        
        except Exception as e:
            raise e