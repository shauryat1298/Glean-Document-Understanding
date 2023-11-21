from collections import Counter
import warnings
from transformers import LayoutLMv3Tokenizer
from src.Glean.utils import str_utils

class VocabularyBuilder():
    """Vocabulary builder class to generate vocabulary."""
    
    def __init__(self, max_size = 512):
        self._words_counter = Counter()
        self.max_size = max_size
        self._vocabulary = { '<PAD>':0, '<NUMBER>':1, '<RARE>':2 }
        self.built = False

        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base-uncased")
        self.layoutlmv3_vocab = self.tokenizer.get_vocab()
        
    def add(self, word):
        if not str_utils.is_number(word):
            self._words_counter.update([word.lower()])
            
    def load_layoutlmv3_vocabulary(self):
        for token, token_id in self.layoutlmv3_vocab.items():
            self._vocabulary[token] = token_id + len(self._vocabulary)

        print(f"LayoutLMv3 Vocabulary loaded. Size: {len(self._vocabulary)}")
        self.built = True

    def build(self):
        if not self.built:
            warnings.warn(
                "The vocabulary is not built. Use VocabularyBuilder.load_layoutlmv3_vocabulary() or add words before building. Returning default vocabulary.", Warning)
        return self._vocabulary

    def get_vocab(self):
        if not self.built:
            warnings.warn(
                "The vocabulary is not built. Use VocabularyBuilder.build(). Returning default vocabulary.", Warning)
            return self._vocabulary
        else:
            return self._vocabulary

