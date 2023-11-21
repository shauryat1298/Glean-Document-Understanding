"""Module with positional and word embeddings for candidates."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LayoutLMv3Model

class NeighbourEmbedding(nn.Module):

    def __init__(self, vocab_size, dimension):
        super().__init__()
        self.lm_model = LayoutLMv3Model.from_pretrained('nielsr/layoutlmv3-finetuned-funsd')
        self.word_embed = nn.Linear(768, dimension)
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, dimension)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, words, positions):

        # Word embedding
        # print(words.shape)
        embedding = self.lm_model.embeddings.word_embeddings(words) # self.word_embed(words)
        # print("Embedding shape: ", embedding.shape )
        embedding = self.word_embed(embedding)
        # print("Embedding shape: ", embedding.shape )
        # Position embedding
        pos = F.relu(self.linear1(positions))
        pos = self.dropout1(pos)
        pos = F.relu(self.linear2(pos))
        pos = self.dropout1(pos)

        # Concatenating word and position embeddings
        neighbour_embedding = torch.cat((embedding, pos), dim=2)

        return neighbour_embedding
