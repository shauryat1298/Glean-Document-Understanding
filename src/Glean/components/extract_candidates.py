import re
import os
import json
from pathlib import Path
from src.Glean import logger
from src.Glean.entity.config_entity import ExtractCandidatesConfig

class ExtractCandidates:
    def __init__(self, config:ExtractCandidatesConfig):
        self.config = config
    
    def get_reg_nums(self, all_words):
        self.reg_nums = []
        reg_no_re = r'^[0-9a-zA-Z-:]+$'
        for word in all_words:
            if not re.search('\d', word['text']):
                continue
            if len(word['text']) < 4:
                continue
            result = re.findall(reg_no_re, word['text'])
            if result:
                self.reg_nums.append({
                    'text': word['text'],
                    'x1': word['x1'],
                    'y1': word['y1'],
                    'x2': word['x2'],
                    'y2': word['y2']
                })

        return self.reg_nums
    
    def get_candidates(self, data):
        self.all_words = []  

        element_width = int(data['ocr']['pages'][0]['dimension']['width'])
        element_height = int(data['ocr']['pages'][0]['dimension']['height'])

        for token in data['ocr']['pages'][0]['tokens']:
            if token['text'].strip() != "":
                self.all_words.append({
                    'text': token['text'],
                    'x1': int(round(float(token['bbox'][1])*element_width)),
                    'y1': int(round(float(token['bbox'][2])*element_height)),
                    'x2': int(round(float(token['bbox'][3])*element_width)),
                    'y2': int(round(float(token['bbox'][4])*element_height))})
        text = ' '.join([word['text'].strip() for word in self.all_words])

        try:
            reg_num_candidates = self.get_reg_nums(self.all_words)
        except Exception as e:
            reg_num_candidates = []
        # try:
        #     reg_name_candidates = get_reg_names(data)
        # except Exception as e:
        #     reg_name_candidates = []

        self.candidate_data = {
            'registration_num': reg_num_candidates
            # 'registrant_name': reg_name_candidates
            # 'total': total_amount_candidates
        }
        return self.candidate_data
    
    def candidates_for_all_ocr(self):
        ocr_path = Path(self.config.ocr_dir)
        output_dir = Path(self.config.candidates_dir)

        annotation_files = list(ocr_path.glob("*.json"))

        for ann in annotation_files:
            with open(ann, 'r', encoding='utf-8') as f:
                data = json.load(f)
                result = self.get_candidates(data)

                output_file_name = os.path.splitext(data['filename'])[0]
                output_file_path = os.path.join(output_dir, output_file_name+'.json')

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    json.dump(result, output_file, ensure_ascii=False, indent=2)
        
        logger.info(f"Candidates saved to {output_dir}")