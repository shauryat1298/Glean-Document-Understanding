from src.Glean.entity.config_entity import EvaluationConfig
from src.Glean.components.train import TrainModel
from src.Glean.config.configuration import ConfigurationManager
from network import dataset
from pathlib import Path
import torch
from src.Glean import logger
from torch.utils import data
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from sklearn.metrics import recall_score, precision_score, f1_score

class Evaluation:
    def __init__(self, config:EvaluationConfig):
        self.config = config

    def evaluate(self, model, val_dataloader, criterion):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        val_accuracy = 0.0
        val_loss = 0.0
        y_preds = []
        y_labels = []

        with torch.no_grad():
            for val_field, val_candidate, val_words, val_positions, masks, val_labels in val_dataloader:
                val_field = val_field.to(device)
                val_candidate = val_candidate.to(device)
                val_words = val_words.to(device)
                val_positions = val_positions.to(device)
                masks = masks.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_field, val_candidate, val_words, val_positions, masks)
                validation_loss = criterion(val_outputs, val_labels)

                val_preds = val_outputs.round()
                y_preds.extend(list(val_preds.cpu().detach().numpy().reshape(1, -1)[0]))
                y_labels.extend(list(val_labels.cpu().detach().numpy().reshape(1, -1)[0]))

                val_accuracy += torch.sum(val_preds == val_labels).item()
                val_loss += validation_loss.item()

            val_loss = val_loss / val_dataloader.sampler.num_samples
            val_accuracy = val_accuracy / val_dataloader.sampler.num_samples
            recall = recall_score(y_labels, y_preds)
            precision = precision_score(y_labels, y_preds)
            f1 = f1_score(y_labels, y_preds)

        return val_accuracy, val_loss, recall, precision, f1
    
    def evaluate_model(self):
        config = ConfigurationManager()
        train_config = config.train_model_config()

        doc_data = dataset.DocumentsDataset(config=train_config, split_name='val')

        test_data = data.DataLoader(doc_data, batch_size=train_config.batch_size, shuffle=True)

        # rlie = Model(VOCAB_SIZE, config.EMBEDDING_SIZE, config.NEIGHBOURS, config.HEADS)
        criterion = torch.nn.BCELoss()
        relie = torch.load(Path(self.config.best_model_dir)/"model.pth")
        # criterion = FocalLoss(alpha=2, gamma=5)

        test_accuracy, test_loss, test_recall, test_precision, test_f1 = self.evaluate(relie, test_data, criterion)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "Test Accuracy": test_accuracy,
                "Test Loss": test_loss,
                "Test Recall": test_recall,
                "Test Precision": test_precision,
                "Test F1": test_f1
            })
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(relie, "model", registered_model_name="TransformerModel")
            else:
                mlflow.pytorch.log_model(relie, "model")
        logger.info(f"Test Accuracy: {test_accuracy} Test Loss: {test_loss} Test Recall: {test_recall} Test Precision {test_precision} Test F1 {test_f1}")