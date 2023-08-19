from django.contrib import admin
from django.urls import path, include
from .views import UploadViewSet, PlotAPIView

urlpatterns = [
    path('', UploadViewSet.as_view({'get': 'list', 'post':'upload'})),
    path('plot/', PlotAPIView.as_view())
]