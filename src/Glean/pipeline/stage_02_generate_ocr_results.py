from src.Glean.config.configuration import ConfigurationManager
from src.Glean.components.generate_ocr_results import GenerateOCRResults
from src.Glean import logger

STAGE_NAME = "Generate OCR Results"

class GenerateOCRResultsPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        generate_ocr_results_config = config.generate_ocr_results_config()
        generate_ocr_results = GenerateOCRResults(generate_ocr_results_config)
        generate_ocr_results.read_and_cut()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = GenerateOCRResultsPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e