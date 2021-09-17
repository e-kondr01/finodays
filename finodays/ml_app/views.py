from rest_framework import views
from rest_framework.response import Response

from finodays.ml_app.logreg import LogRegClassifier


class PredictView(views.APIView):
    """View для предсказания настроения сообщения"""

    def post(self, request, *args, **kwargs):

        classifier = LogRegClassifier()

        prediction = classifier.compute_prediction(request.data)
        response = {
            "настроение клиента": prediction
        }
        response["совет"] = classifier.get_advice()

        return Response(response)
