from src.Glean.config.configuration import ConfigurationManager
from src.Glean.components.prepare_split_full import PrepareSplit
from src.Glean import logger

STAGE_NAME = "Prepare Split"

class PrepareSplitPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_split_config = config.prepare_split_config()
        prepare_split = PrepareSplit(prepare_split_config)
        prepare_split.split_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareSplitPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e