from django.urls import path
from serializer import views

from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('Scans/', views.Scan_list),
    path('Scans/<int:pk>/', views.Scan_detail),
    path('Plant_Infos/', views.Plant_Info_list),
    path('Plant_Infos/<int:pk>/', views.Plant_Info_detail),
]


urlpatterns = format_suffix_patterns(urlpatterns)
