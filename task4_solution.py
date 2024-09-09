import json
import pickle

import numpy as np
from scipy.sparse import vstack


class Solution:
    def __init__(self):
        with open("dev-dataset-task2023-04.json") as f:
            self.dev_dataset = json.load(f)
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        self.train_x = self.tokenizer.transform(
            np.array([text for text, _ in self.dev_dataset])
        )
        self.train_y = np.array([int(label) for _, label in self.dev_dataset])

    def predict(self, text: str) -> str:
        tokenized_text = self.tokenizer.transform([text])
        pred = self.model.predict(tokenized_text)
        # Update model with already processed data
        self.train_x = vstack([self.train_x, tokenized_text])
        self.train_y = np.concatenate([self.train_y, pred])
        self.model.fit(self.train_x, self.train_y)
        return str(int(pred[0]))
