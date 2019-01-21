import json
from tensorflow.keras.preprocessing.text import text_to_word_sequence

class SimpleTokenizer:
    """A simple tokenizer to load the dictionary from a json file.
    """
    def __init__(self, dictionary_file, max_words=10000):
        """Initialize SimpleTokenizer
        
        Args:
            dictionary_file: a json file storing the dictionary.
        """
        self.dictionary_file = dictionary_file
        with open(self.dictionary_file, "r") as f:
            self.word_index = json.load(f)
        self.word_index_reverse = {v:k for k,v in self.word_index.items()}

        self.max_words = max_words
    
    def texts_to_sequences(self, texts):
        """mimic the texts_to_sequences() function in keras' Tokenizer
        Args:
            texts: a list of strings
        """
        seq = []
        for line in texts:
            s = []
            for w in text_to_word_sequence(line):
                if w in self.word_index:
                    if self.word_index[w] < self.max_words:
                        s.append(self.word_index[w])
            seq.append(s)
        return seq