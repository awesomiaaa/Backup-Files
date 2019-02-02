"""restapi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
import serializer
import ml


from rest_framework import routers

from serializer.views import ItemViewSet, Plant_Infos, Users, Plant_Lists

from django.conf.urls.static import static
from django.conf import settings

router = routers.DefaultRouter()
router.register(r'Scans', ItemViewSet, base_name='Scans')
router.register(r'Plant_Infos', Plant_Infos, base_name='Plant_Infos')
router.register(r'Users', Users, base_name="Users")
router.register(r'Plant_Lists', Plant_Lists, base_name="Plant_Lists")

urlpatterns = [
    path('admin/', admin.site.urls),
##    path('',include('serializer.urls')),
    path('',include(router.urls)),
    path('start/',include('ml.urls')),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
