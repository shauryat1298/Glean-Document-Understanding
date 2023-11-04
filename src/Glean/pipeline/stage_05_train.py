from src.Glean.config.configuration import ConfigurationManager
from src.Glean.components.train import TrainModel
from src.Glean import logger

STAGE_NAME = "Train Model"

class TrainModelPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        train_model_config = config.train_model_config()
        train_model = TrainModel(train_model_config)
        train_model.train_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e