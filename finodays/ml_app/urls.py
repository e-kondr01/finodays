from django.conf.urls import url

from finodays.ml_app.views import *


urlpatterns = [
    url("predict", PredictView.as_view()),
]
