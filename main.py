from src.Glean import logger
from src.Glean.pipeline.stage_01_prepare_split import PrepareSplitPipeline
from src.Glean.pipeline.stage_02_generate_ocr_results import GenerateOCRResultsPipeline
from src.Glean.pipeline.stage_04_ground_truth_annotations import GroundTruthAnnotationsPipeline
from src.Glean.pipeline.stage_03_extract_candidates import ExtractCandidatesPipeline
from src.Glean.pipeline.stage_05_train import TrainModelPipeline

# STAGE_NAME = "Prepare Split"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = PrepareSplitPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Generate OCR Results"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = GenerateOCRResultsPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Ground Truth Annotations"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = GroundTruthAnnotationsPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Extract Candidates"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = ExtractCandidatesPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Train Model"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TrainModelPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
except Exception as e:
    logger.exception(e)
    raise e
