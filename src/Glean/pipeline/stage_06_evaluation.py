from src.Glean.config.configuration import ConfigurationManager
from src.Glean.components.model_evaluation import Evaluation
from src.Glean import logger

STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation.evaluate_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e