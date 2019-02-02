from django.db import models

# Create your models here.

from .storage import OverwriteStorage

class Scan(models.Model):
    status = models.BooleanField(default=True)
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
    	ordering = ('date',)

class Plant_Info(models.Model):
    scan_no = models.ForeignKey(Scan,related_name='scan_details', on_delete=models.CASCADE)
    plant_no = models.IntegerField()
    condition = models.CharField(max_length=50)
    disease = models.TextField()
    diagnosis = models.TextField()
    model_pic = models.ImageField(storage=OverwriteStorage(), upload_to = 'restapi/imagemodel', default="restapi/imagemodel")
    
    class Meta:
    	ordering = ('scan_no',)

class Users(models.Model):
    name = models.CharField(max_length=20)
    email = models.EmailField()
    password = models.CharField(max_length=20)

class Plant_List(models.Model):
    plant_name = models.CharField(max_length=20)
    specific_plant = models.CharField(max_length=20)
    plant_width = models.IntegerField()
    plot_size = models.IntegerField()
    plant_distance = models.IntegerField()
