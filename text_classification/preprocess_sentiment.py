import pandas as pd

from preprocess import preprocess_text_en
from tqdm import tqdm

"""Препроцессинг текста из датасета sentiment140"""

tqdm.pandas()

orig_df = pd.read_csv("sentiment140.csv", encoding="latin-1",
                      names=["label", "B", "C", "D", "E", "text"])
df = orig_df[["label", "text"]]
df["text"] = df["text"].progress_apply(preprocess_text_en)
df.dropna(subset=["text"], inplace=True)
df.to_csv("preprocessed_sentiment.csv")
