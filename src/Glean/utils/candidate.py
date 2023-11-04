import json
import traceback
from src.Glean.utils import operations as op
from tqdm import tqdm
from pathlib import Path
from src.Glean import logger

def attach_candidate(annotation, candidate_path):

    for anno in tqdm(annotation, desc='Attaching Candidate'):
        try:
            file_name = anno['filename']
            candidate_json = Path(candidate_path) / (file_name + ".json")
            with open(candidate_json, 'r') as f:
                candidate_data = json.load(f)

            for cls, cads in candidate_data.items():

                for true_cad in anno['fields'][cls]['true_candidates']:
                    for cad in cads:
                        iou = op.bb_intersection_over_union([true_cad['x1'], true_cad['y1'],
                                                             true_cad['x2'], true_cad['y2']],
                                                            [cad['x1'], cad['y1'], cad['x2'], cad['y2']])
                        if iou > 0.5:
                            anno['fields'][cls]['true_candidates'].remove(true_cad)
                            anno['fields'][cls]['true_candidates'].append(cad)
                            candidate_data[cls].remove(cad)
                            break
                anno['fields'][cls]['other_candidates'] = candidate_data[cls]

        except Exception:
            trace = traceback.format_exc()
            logger.info("Error in processing candidate: %s : %s" % (anno['filename'], trace))
            break

    return annotation

# if __name__ == '__main__':
#     annotation, classes_count, class_mapping = annotation_parser.get_data(config.XML_DIR, "train")
#     print(annotation[0])
#     annotation = attach_candidate(annotation, config.CANDIDATE_DIR)
#     print(annotation[0])
