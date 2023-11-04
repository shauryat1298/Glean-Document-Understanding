from src.Glean.config.configuration import ConfigurationManager
from src.Glean.components.extract_candidates import ExtractCandidates
from src.Glean import logger

STAGE_NAME = "Extract Candidates"

class ExtractCandidatesPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        extract_candidates_config = config.extract_candidates_config()
        extract_candidates = ExtractCandidates(extract_candidates_config)
        extract_candidates.candidates_for_all_ocr()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ExtractCandidatesPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e