from src.Glean.config.configuration import ConfigurationManager
from src.Glean.components.ground_truth_annotations import GroundTruthAnnotations
from src.Glean import logger

STAGE_NAME = "Ground Truth Annotations"

class GroundTruthAnnotationsPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        ground_truth_annotations_config = config.ground_truth_annotations_config()
        ground_truth_annotations = GroundTruthAnnotations(ground_truth_annotations_config)
        ground_truth_annotations.read_and_cut()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = GroundTruthAnnotationsPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e