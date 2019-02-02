from django.contrib import admin

# Register your models here.

from .models import Scan, Plant_Info


admin.site.register(Scan)
admin.site.register(Plant_Info)
