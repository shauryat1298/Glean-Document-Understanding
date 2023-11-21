import torch
from torch.utils import data
from network.model import Model
from network import dataset
from src.Glean.utils.evaluate import evaluate
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.Glean import logger
from src.Glean.entity.config_entity import TrainModelConfig
from pathlib import Path
import os
import pandas as pd

class TrainModel:
    def __init__(self, config:TrainModelConfig):
        self.config = config
    
    def train(self, model, train_dataloader, val_dataloader, epochs):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        # writer = SummaryWriter(comment=f"LR_{self.config.lr}_BATCH_{self.config.batch_size}")
        # criterion = nn.BCELoss()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        train_loss_history = []
        train_accuracy_history = []
        recall_history = []
        precision_history = []
        f1_history = []
        val_loss_history = []
        val_accuracy_history = []
        val_recall_history = []
        val_precision_history = []
        val_f1_history = []
        val_max_score = 0.0

        for epoch in range(1, epochs + 1):

            train_loss = 0.0
            train_accuracy = 0.0
            y_preds = []
            y_labels = []

            for field, candidate, words, positions, masks, labels in tqdm(train_dataloader, desc="Epoch %s" % epoch):
                # print(field.dim())
                field = field.to(device)
                candidate = candidate.to(device)
                words = words.to(device)
                positions = positions.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                outputs = model(field, candidate, words, positions, masks)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = outputs.round()
                y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
                y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))

                train_accuracy += torch.sum(preds == labels).item()
                train_loss += loss.item()

            else:
                val_accuracy, val_loss, val_recall, val_precision, val_f1 = evaluate(model, val_dataloader, criterion)

                train_loss = train_loss / train_dataloader.sampler.num_samples
                train_accuracy = train_accuracy / train_dataloader.sampler.num_samples
                recall = recall_score(y_labels, y_preds)
                precision = precision_score(y_labels, y_preds)
                f1score = f1_score(y_labels, y_preds)

                train_loss_history.append(train_loss)
                train_accuracy_history.append(train_accuracy)
                recall_history.append(recall)
                precision_history.append(precision)
                f1_history.append(f1score)
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_accuracy)
                val_recall_history.append(val_recall)
                val_precision_history.append(val_precision)
                val_f1_history.append(val_f1)

                if val_recall > val_max_score: # Saving the best model
                    print('saving model....')
                    val_max_score = val_recall
                    os.makedirs(Path(self.config.best_model_dir), exist_ok=True)
                    torch.save(model, Path(self.config.best_model_dir)/'model.pth')
                print(f"Metrics for Epoch {epoch}:  Loss:{round(train_loss, 4)} \
                        Recall: {round(recall, 4)} \
                        Validation Loss: {round(val_loss, 4)} \
                        Validation Recall: {round(val_recall, 4)}")

        return {
            # 'training_loss': train_loss_history,
            # 'training_accuracy': train_accuracy_history,
            'training_recall': recall_history,
            'training_precision': precision_history,
            'training_f1': f1_history,
            # 'validation_loss': val_loss_history,
            # 'validation_accuracy': val_accuracy_history,
            'validation_recall': val_recall_history,
            'validation_precision': val_precision_history,
            'validation_f1': val_f1_history
        }
    
    def train_model(self):
        train_data = dataset.DocumentsDataset(self.config, 'train')
        val_data = dataset.DocumentsDataset(self.config, 'val')

        VOCAB_SIZE = len(train_data.vocab)

        training_data = data.DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        validation_data = data.DataLoader(val_data, batch_size=self.config.batch_size, shuffle=True)

        relie = Model(VOCAB_SIZE, self.config.embedding_size, self.config.neighbours, self.config.heads)

        history = self.train(relie, training_data, validation_data, self.config.epochs)
        logger.info(pd.DataFrame(history))

