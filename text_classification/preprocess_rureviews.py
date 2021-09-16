import pandas as pd

from preprocess import preprocess_text_ru
from tqdm import tqdm

"""Препроцессинг текста из датасета rureviews"""

tqdm.pandas()

df = pd.read_csv("rureviews.csv", skiprows=1,
                 names=("review", "sentiment"),
                 delimiter="	")
df["review"] = df["review"].progress_apply(preprocess_text_ru)
df.dropna(subset=["review"], inplace=True)
df.to_csv("preprocessed_rureviews.csv")
