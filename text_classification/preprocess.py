import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from typing import List


def remove_chars_from_text(text: str) -> str:
    """Удаляет символы из данного списка из текста"""
    chars = string.punctuation + string.digits + string.ascii_lowercase
    res = ""
    for i in range(len(text)):
        if text[i] in chars:
            try:
                if text[i+1] != " ":
                    res += " "
            except IndexError:
                pass
        else:
            res += text[i]
    return res


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


def preprocess_text_en(text: str) -> str:
    """Осуществляет препроцессинг английского текста"""
    text = text.lower()
    text = remove_chars_from_text(text)

    text_tokens = word_tokenize(text)

    stop_words = stopwords.words("english")
    text_tokens = [
        word for word in text_tokens if word not in stop_words
    ]

    text_tokens = stem_text(text_tokens)
    text_tokens = lemmatize_en(text_tokens)

    return " ".join(text_tokens)


def preprocess_text_ru(text: str) -> str:
    """Осуществляет препроцессинг русского текста"""
    text = text.lower()
    text = remove_chars_from_text(text)

    text_tokens = word_tokenize(text)

    stop_words = stopwords.words("russian")
    stop_words.remove("нет")
    stop_words.remove("не")
    stop_words.remove("ни")
    stop_words.remove("никогда")
    stop_words.remove("нельзя")
    stop_words.remove("хорошо")
    stop_words.remove("ничего")
    stop_words.remove("больше")
    stop_words.remove("лучше")

    text_tokens = [
        word for word in text_tokens if word not in stop_words
    ]

    text_tokens = lemmatize_ru(text_tokens)

    return " ".join(text_tokens)
