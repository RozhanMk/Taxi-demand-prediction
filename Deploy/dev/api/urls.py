from django.contrib import admin
from django.urls import path, include
from .views import UploadViewSet

urlpatterns = [
    path('predict/', UploadViewSet.as_view({'get': 'list', 'put':'upload'})),
]