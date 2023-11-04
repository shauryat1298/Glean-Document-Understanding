from pathlib import Path
import traceback
from tqdm import tqdm
from src.Glean.entity.config_entity import TrainModelConfig
import json

def get_data(config:TrainModelConfig, split_name='train'):

    annotations = []
    classes_count = {}
    class_mapping = {}

    split_file_path = Path(config.split_dir) / f"{split_name}.txt"
    print(split_file_path)
    with open(split_file_path, 'r') as f:
        valid_files = f.read().split("\n")
    annotation_files = list(Path(config.ground_truth_dir).glob("*.json"))
    annotation_files = [an for an in annotation_files if an.stem in valid_files]

    for annot in tqdm(annotation_files, desc="Reading Annotations"):
        try:
            with open(annot, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
            element_width = int(data['ocr']['pages'][0]['dimension']['width'])
            element_height = int(data['ocr']['pages'][0]['dimension']['height'])

            annotation_data = {'filename': annot.stem, 'width': element_width,
                               'height': element_height, 'fields': {'registration_num': {'true_candidates': [],
                                                                                   'other_candidates': []}
                                                                    # 'registrant_name': {'true_candidates': [],
                                                                    #                  'other_candidates': []}
                                                                    }}
            for i, cls in enumerate(annotation_data['fields']):
                if cls not in classes_count:
                    classes_count[cls] = 0
                    class_mapping[cls] = i
            
            for ann in data["annotations"]:
                class_name = ann[0]
                # class_name = ann[1][0][1][1]
                # print(class_name)
                if class_name not in annotation_data['fields']:
                    print("Unidentified field Found:", class_name, "in file:", annot.name)
                    continue
                else:
                    classes_count[class_name] += 1
                
                x1 = int(round(float(ann[1][0][1][1])*element_width))
                y1 = int(round(float(ann[1][0][1][2])*element_height))
                x2 = int(round(float(ann[1][0][1][3])*element_width))
                y2 = int(round(float(ann[1][0][1][4])*element_height))

                annotation_data['fields'][class_name]['true_candidates'].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
            
            annotations.append(annotation_data)
        
        except Exception:
            print(traceback.format_exc())
    
    return annotations, classes_count, class_mapping

# if __name__ == '__main__':
#     xml_path = config.XML_DIR
#     print(xml_path)
#     k, l, m = get_data(xml_path)
#     print(len(k), l, m)