import joblib
import pandas as pd

from django.conf import settings
from finodays.ml_app.classifiers.classifier import Classifier


class LogRegClassifier(Classifier):
    def __init__(self):
        self.model = joblib.load(settings.APPS_DIR / "ml_app" / "logreg.joblib")

    def preprocessing(self, input_data):
        return pd.DataFrame(input_data, index=[0])["text"]

    def predict(self, input_data):
        return self.model.predict_proba(input_data)[0]

    def postprocessing(self, input_data):
        classes = self.model.classes_
        return [{classes[i]: input_data[i]} for i in range(len(input_data))]

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
