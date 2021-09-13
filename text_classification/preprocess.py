import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from typing import List


def remove_chars_from_text(text: str, chars: List[str]) -> str:
    """Удаляет символы из данного списка из текста"""
    return "".join([ch for ch in text if ch not in chars])


def lemmatize_ru(tokens: List[str]) -> List[str]:
    """Лемматизация текста"""
    res = []
    morph = MorphAnalyzer()
    for token in tokens:
        token = morph.normal_forms(token)[0]
        res.append(token)
    return res


def stem_text(tokens: List[str]) -> List[str]:
    """Стеммизация текста - англ"""
    stemmer = PorterStemmer()
    res = []
    for token in tokens:
        res.append(stemmer.stem(token))
    return res


def lemmatize_en(tokens: List[str]) -> List[str]:
    """Лемматизация текста - англ"""
    lemmatizer = WordNetLemmatizer()
    res = []
    for token in tokens:
        res.append(lemmatizer.lemmatize(token))
    return res


def preprocess_text(text: str, ru: bool = True):
    """Осуществляет препроцессинг текста"""
    text = text.lower()
    spec_chars = string.punctuation + '\n\xa0«»\t—…'
    text = remove_chars_from_text(text, spec_chars)
    text = remove_chars_from_text(text, string.digits)

    text_tokens = word_tokenize(text)

    # TODO: добавить специфические для банка stop words
    if ru:
        stop_words = stopwords.words("russian")
    else:
        stop_words = stopwords.words("english")
    text_tokens = [
        word for word in text_tokens if word not in stop_words
    ]

    if ru:
        text_tokens = lemmatize_ru(text_tokens)
    else:
        text_tokens = stem_text(text_tokens)
        text_tokens = lemmatize_en(text_tokens)

    return text_tokens
