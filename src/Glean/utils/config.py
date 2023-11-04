from pathlib import Path


NEIGHBOURS = 5
HEADS = 4
EMBEDDING_SIZE = 16
VOCAB_SIZE = 1000
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.001

current_directory = Path("C:/Users/shaur/Desktop/Glean_Implementation/utils")

XML_DIR = current_directory / "dataset" / "xmls"
OCR_DIR = current_directory / "dataset" / "ocr_results"
IMAGE_DIR = current_directory / "dataset" / "images" / "dataset.jsonl.gz"
TRAIN_TEST_SPLIT = current_directory / "dataset" / "images" / "FARA-lv1-single_Short-Form-train_200-test_300-valid_100-SD_2.json"
CANDIDATE_DIR = current_directory / "dataset" / "candidates"
SPLIT_DIR = current_directory / "dataset" / "split"
OUTPUT_DIR = current_directory / "output"

# INFERENCE
image_test_dir = Path("C:/Users/shaur/Desktop/Glean_Implementation/vrdu-main/vrdu-main/registration-form/main/pngs")
IMAGE_OCR_PATH = current_directory / "dataset" / "ocr_results" / "20180105_Hill and Knowlton Strategies, LLC_Bensaid, Noelle_Short-Form.json"
IMAGE_PATH = image_test_dir / "20180105_Hill and Knowlton Strategies, LLC_Bensaid, Noelle_Short-Form_1.png"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)