import sys
from pathlib import Path
path = Path("C:/Users/shaur/Desktop/Glean_Implementation")
sys.path.append(str(path))

import torch
from utils import Neighbour, config, preprocess
import cv2
import traceback
import numpy as np
import argparse
import os
import re
import json
import pickle
import importlib
importlib.reload(config)



def load_saved_vocab(path):
    cached_data = pickle.load(open(path, 'rb'))
    return cached_data['vocab'], cached_data['mapping']


def parse_input(annotations, fields_dict, n_neighbours=5, vocabulary=None):
    """Generates input samples from annotations data."""
    field_ids = list()
    candidate_cords = list()
    neighbours = list()
    neighbour_cords = list()
    n_classes = len(fields_dict)
    for field, value in annotations.items():
        if annotations[field]:
            for idx, val in enumerate(value):
                _neighbours, _neighbour_cords = preprocess.get_neighbours(
                    val['neighbours'],
                    vocabulary, n_neighbours
                )
                field_ids.append(np.eye(n_classes)[fields_dict[field]])
                candidate_cords.append(
                    [
                        val['x'],
                        val['y']
                    ]
                )
                neighbours.append(_neighbours)
                neighbour_cords.append(_neighbour_cords)
    return torch.Tensor(field_ids).type(torch.FloatTensor), torch.Tensor(candidate_cords).type(
        torch.FloatTensor), torch.Tensor(neighbours).type(torch.int64), torch.Tensor(neighbour_cords).type(
        torch.FloatTensor)


def normalize_coordinates(annotations, width, height):
    try:
        for cls, cads in annotations.items():
            for i, cd in enumerate(cads):
                cd = cd.copy()
                x1 = cd['x1']
                y1 = cd['y1']
                x2 = cd['x2']
                y2 = cd['y2']
                cd['x'] = ((x1 + x2) / 2) / width
                cd['y'] = ((y1 + y2) / 2) / height
                neighbours = []
                for neh in cd['neighbours']:
                    neh = neh.copy()
                    x1_neh = neh['x1']
                    y1_neh = neh['y1']
                    x2_neh = neh['x2']
                    y2_neh = neh['y2']
                    # calculating neighbour position w.r.t candidate
                    neh['x'] = (((x1_neh + x2_neh) / 2) / width) - cd['x']
                    neh['y'] = (((y1_neh + y2_neh) / 2) / height) - cd['y']
                    neighbours.append(neh)
                cd['neighbours'] = neighbours
                annotations[cls][i] = cd
    except Exception:
        trace = traceback.format_exc()
        print("Error in normalizing position: %s : %s" % (trace, trace))
    return annotations


def get_reg_nums(all_words):
    reg_nums = []
    reg_no_re = r'^[0-9a-zA-Z-:]+$'
    for word in all_words:
        if not re.search('\d', word['text']):
            continue
        if len(word['text']) < 4:
            continue
        result = re.findall(reg_no_re, word['text'])
        if result:
            reg_nums.append({
                'text': word['text'],
                'x1': word['x1'],
                'y1': word['y1'],
                'x2': word['x2'],
                'y2': word['y2']
            })

    return reg_nums

def get_candidates(data):
        all_words = []  

        element_width = int(data['ocr']['pages'][0]['dimension']['width'])
        element_height = int(data['ocr']['pages'][0]['dimension']['height'])

        for token in data['ocr']['pages'][0]['tokens']:
            if token['text'].strip() != "":
                all_words.append({
                    'text': token['text'],
                    'x1': int(round(float(token['bbox'][1])*element_width)),
                    'y1': int(round(float(token['bbox'][2])*element_height)),
                    'x2': int(round(float(token['bbox'][3])*element_width)),
                    'y2': int(round(float(token['bbox'][4])*element_height))})
        text = ' '.join([word['text'].strip() for word in all_words])

        try:
            reg_num_candidates = get_reg_nums(all_words)
        except Exception as e:
            reg_num_candidates = []
        # try:
        #     reg_name_candidates = get_reg_names(data)
        # except Exception as e:
        #     reg_name_candidates = []

        candidate_data = {
            'registration_num': reg_num_candidates
            # 'registrant_name': reg_name_candidates
            # 'total': total_amount_candidates
        }
        return candidate_data

def attach_neighbour(width, height, data, candidates):
            
    words = []

    for token in data['ocr']['pages'][0]['tokens']:
        if token['text'].strip() != "":
            words.append({
                'text': token['text'],
                'x1': int(round(float(token['bbox'][1])*width)),
                'y1': int(round(float(token['bbox'][2])*height)),
                'x2': int(round(float(token['bbox'][3])*width)),
                'y2': int(round(float(token['bbox'][4])*height))})
    
    x_offset = int(width * 0.1)
    y_offset = int(height * 0.1)

    for cls, both_cads in candidates.items():
        for cad in both_cads:
            neighbours = Neighbour.find_neighbour(cad, words, x_offset, y_offset, width, height)
            cad['neighbours'] = neighbours
    return candidates



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = cv2.imread(str(config.IMAGE_PATH))
    height, width, _ = image.shape

    with open(config.IMAGE_OCR_PATH, encoding='utf-8') as f:
        ocr_results = json.load(f)
    vocab, class_mapping = load_saved_vocab(r"C:\Users\shaur\Desktop\Glean_Implementation\utils\output\cached_data_val.pickle")
    candidates = get_candidates(ocr_results)
    candidates_with_neighbours = attach_neighbour(width, height, ocr_results, candidates)

    annotation = normalize_coordinates(candidates_with_neighbours, width, height)
    _data = parse_input(annotation, class_mapping, config.NEIGHBOURS, vocab)
    field_ids, candidate_cords, neighbours, neighbour_cords = _data
    rlie = torch.load(r"C:\Users\shaur\Desktop\Glean_Implementation\utils\output\model.pth")
    rlie = rlie.to(device)
    field_ids = field_ids.to(device)
    candidate_cords = candidate_cords.to(device)
    neighbours = neighbours.to(device)
    neighbour_cords = neighbour_cords.to(device)
    field_idx_candidate = np.argmax(field_ids.detach().to('cpu').numpy(), axis=1)
    with torch.no_grad():
        rlie.eval()
        val_outputs = rlie(field_ids, candidate_cords, neighbours, neighbour_cords, None)
    val_outputs = val_outputs.to('cpu').numpy()
    out = {cl: np.argmax(val_outputs[np.where(field_idx_candidate == cl)]) for cl in np.unique(field_idx_candidate)}
    true_candidate_color = (0, 255, 0)
    output_candidates = {}
    output_image = image.copy()
    for idx, (key, value) in enumerate(candidates.items()):
        if idx in out:
            candidate_idx = out[idx]
            cand = candidates[key][candidate_idx]
            output_candidates[key] = cand['text']
            cand_coords = [cand['x1'], cand['y1'], cand['x2'], cand['y2']]
            cv2.rectangle(output_image, (cand_coords[0], cand_coords[1]), (cand_coords[2], cand_coords[3]),
                            true_candidate_color, 5)
            
    cv2.imshow("Predicted Candidates", output_image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print(output_candidates)
    return output_candidates


if __name__ == '__main__':
    main()
