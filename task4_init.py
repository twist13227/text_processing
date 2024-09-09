import json
import pickle
import numpy as np

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor

nltk.download("stopwords")

with open("dev-dataset-task2023-04.json", encoding="utf-8") as f:
    dev_dataset = json.load(f)

model = KNeighborsRegressor(n_neighbors=1, metric="cosine", weights="uniform")
tokenizer = TfidfVectorizer(min_df=3, stop_words=nltk.corpus.stopwords.words("russian"))

train_x = tokenizer.fit_transform(np.array([text for text, _ in dev_dataset]))
train_y = np.array([int(label) for _, label in dev_dataset])

model.fit(train_x, train_y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
