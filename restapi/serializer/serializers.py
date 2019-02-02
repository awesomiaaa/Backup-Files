from rest_framework import serializers
from .models import Scan, Plant_Info, Users, Plant_List


##class Scan_Serializer(serializers.ModelSerializer):
##	class Meta:
##		model = Scan
##		fields = ('id', 'status', 'date')
##


class Plant_Info_Serializer(serializers.ModelSerializer):
        class Meta:
                model = Plant_Info
                fields = ('id', 'plant_no', 'condition', 'disease', 'diagnosis','model_pic')


class Scan_Serializer(serializers.ModelSerializer):
        scan_details = Plant_Info_Serializer(many=True, required=False)
        class Meta:
                model = Scan
                fields = ('id','status','date','scan_details',)

class Users_Serializer(serializers.ModelSerializer):
        class Meta:
                model = Users
                fields=('name', 'email', 'password' ,)


class Plant_List_Serializer(serializers.ModelSerializer):
        class Meta:
                model = Plant_List
                fields=('plant_name', 'specific_plant', 'plant_width', 'plot_size', 'plant_distance')

