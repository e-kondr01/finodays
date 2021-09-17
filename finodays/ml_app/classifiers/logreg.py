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
        return self.model.predict(input_data)

    def postprocessing(self, input_data):
        print(input_data)
        return input_data

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
