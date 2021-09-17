from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from finodays.ml_app.views import *


router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet,
                basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    url(r"^ml/", include(router.urls)),
    url("predict", PredictView.as_view()),
    url("test-predict", TestPredictView.as_view())
]
