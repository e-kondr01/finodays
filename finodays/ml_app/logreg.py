import joblib
import pandas as pd

from django.conf import settings

from text_classification.preprocess import preprocess_text_ru


def to_percent(number: float) -> str:
    return str(round(number * 100, 1)) + "%"


class LogRegClassifier():
    """Классификатор сообщения по обученной модели логистической регресии"""

    sentiment = ""

    def __init__(self):
        self.model = joblib.load(
            settings.APPS_DIR / "ml_app" / "logreg.joblib")

    def preprocessing(self, input_data):
        return pd.DataFrame({"text": preprocess_text_ru(input_data["text"])},
                            index=[0])["text"]

    def predict(self, input_data):
        return self.model.predict_proba(input_data)[0]

    def postprocessing(self, input_data):
        classes = self.model.classes_
        return [{classes[i]: input_data[i]} for i in range(len(input_data))]

    def format_prediction(self, prediction):
        formatted = []
        for sentiment in prediction:
            form_sent = {}
            if "neautral" in sentiment.keys():
                form_sent["Нейтральный"] = to_percent(sentiment["neautral"])
                formatted.append(form_sent)
                continue
            elif "negative" in sentiment.keys():
                form_sent["Негативный"] = to_percent(sentiment["negative"])
                formatted.append(form_sent)
                continue
            else:
                form_sent["Позитивный"] = to_percent(sentiment["positive"])
                formatted.append(form_sent)
        return formatted

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
            self.sentiment = list(max(prediction,
                                      key=lambda it: list(
                                          it.values())[0]).keys())[0]
            prediction = self.format_prediction(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction

    def get_advice(self):
        advice = {
            "negative": "Позовите сотрудника по разрешению конфликтов",
            "positive": "Посоветуйте оценить приложение или другой сервис",
            "neautral": "Попросите явно оценить работу",
            "": "ещё не оценено"
        }
        return advice[self.sentiment]
