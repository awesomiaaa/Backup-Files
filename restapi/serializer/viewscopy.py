from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Scan, Plant_Info
from .serializers import Scan_Serializer, Plant_Info_Serializer



@api_view(['GET', 'POST'])
def Scan_list(request, format=None):
    if request.method == 'GET':
        Scans = Scan.objects.all()
        serializer = Scan_Serializer(Scans, many=True)
        return Response({"data":serializer.data})

    elif request.method == 'POST':
        serializer = Scan_Serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(['GET', 'PUT', 'DELETE'])
def Scan_detail(request,pk, format=None):

    try:
        Scans = Scan.objects.get(pk=pk)
    except Scan.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = Scan_Serializer(Scans,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        Plant_Info.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    

@api_view(['GET', 'POST'])
def Plant_Info_list(request, format=None):
    if request.method == 'GET':
        Plant_Infos = Plant_Info.objects.all()
        serializer = Plant_Info_Serializer(Plant_Infos, many=True)
        print(type(serializer.data))
        return Response({'data' : serializer.data})

    elif request.method == 'POST':
        serializer = Plant_Info_Serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(['GET', 'PUT', 'DELETE'])

def Plant_Info_detail(request,pk, format=None):

    try:
        Plant_Infos = Plant_Info.objects.get(pk=pk)
    except Plant_Info.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = Plant_Info_Serializer(Plant_Infos,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        Plant_Info.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    
