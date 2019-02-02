from django.urls import path
from ml import views



urlpatterns = [
    path('', views.camera, name="camera"),
    path('start/', views.start, name="start"),

]

