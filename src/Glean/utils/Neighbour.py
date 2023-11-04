import json
import traceback
from tqdm import tqdm
from pathlib import Path
from src.Glean.utils import operations as op
from src.Glean.utils import vocabulary
from src.Glean import logger

def find_neighbour(cad, words, x_offset, y_offset, width, height):
    # iou_scores = []
    # for w in words:
    #     iou_scores.append(op.bb_intersection_over_union([cad['x1'], cad['y1'], cad['x2'], cad['y2']],
    #                                                     [w['x1'], w['y1'], w['x2'], w['y2']]))

    # if max(iou_scores) > 0.2:
    #     max_ind = iou_scores.index(max(iou_scores))
    #     a['keyword'] = words[max_ind]
    # else:
    #     print("No keyword found in OCR corresponding to: ", str(a), "filename :", file_name)
    #     a['keyword'] = {}

    # neighbour
    words_copy = words.copy()
    if cad in words_copy:
        words_copy.remove(cad)
    neighbours = []

    neighbour_x1 = cad['x1'] - x_offset
    neighbour_x1 = 1 if neighbour_x1 < 1 else neighbour_x1

    neighbour_y1 = cad['y1'] - y_offset
    neighbour_y1 = 1 if neighbour_y1 < 1 else neighbour_y1

    neighbour_x2 = cad['x2'] + x_offset
    neighbour_x2 = width - 1 if neighbour_x2 >= width else neighbour_x2

    neighbour_y2 = cad['y2'] + y_offset
    neighbour_y2 = height - 1 if neighbour_y2 >= height else neighbour_y2

    neighbour_bbox = [neighbour_x1, neighbour_y1, neighbour_x2, neighbour_y2]
    iou_scores = []
    for w in words_copy:
        iou_scores.append(op.bb_intersection_over_boxB(neighbour_bbox, [w['x1'], w['y1'], w['x2'], w['y2']]))

    for i, iou in enumerate(iou_scores):
        if iou > 0.5:
            neighbours.append(words_copy[i])

    return neighbours


def attach_neighbour(annotation, ocr_path, vocab_size):
    
    vocab_builder = vocabulary.VocabularyBuilder(max_size=vocab_size)
    
    for anno in tqdm(annotation, desc="Attaching Neighbours"):
        try:
            file_name = anno['filename']
            ocr_json = Path(ocr_path) / (file_name + ".json")
            with open(ocr_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            element_width = int(data['ocr']['pages'][0]['dimension']['width'])
            element_height = int(data['ocr']['pages'][0]['dimension']['height'])
            
            words = []

            for token in data['ocr']['pages'][0]['tokens']:
                if token['text'].strip() != "":
                    words.append({
                        'text': token['text'],
                        'x1': int(round(float(token['bbox'][1])*element_width)),
                        'y1': int(round(float(token['bbox'][2])*element_height)),
                        'x2': int(round(float(token['bbox'][3])*element_width)),
                        'y2': int(round(float(token['bbox'][4])*element_height))})
                    
                    vocab_builder.add(token['text'])
            
            

            x_offset = int(element_width * 0.1)
            y_offset = int(element_height * 0.1)

            for _, both_cads in anno['fields'].items():
                for cad in both_cads['true_candidates']:
                    neighbours = find_neighbour(cad, words, x_offset, y_offset, element_width, element_height)
                    cad['neighbours'] = neighbours
                for cad in both_cads['other_candidates']:
                    neighbours = find_neighbour(cad, words, x_offset, y_offset, element_width, element_height)
                    cad['neighbours'] = neighbours

        except Exception:
            trace = traceback.format_exc()
            logger.info("Error in finding neighbour: %s : %s" % (anno['filename'], trace))
            break
            
    _vocab = vocab_builder.build()

    return annotation, _vocab

# if __name__ == '__main__':
#     annotation, classes_count, class_mapping = annotation_parser.get_data(config.XML_DIR, "train")
#     print(annotation[0])
#     annotation = candidate.attach_candidate(annotation, config.CANDIDATE_DIR)
#     print(annotation[0])
#     annotation, vocab = attach_neighbour(annotation, config.OCR_DIR, vocab_size=config.VOCAB_SIZE)
#     print(annotation[0])