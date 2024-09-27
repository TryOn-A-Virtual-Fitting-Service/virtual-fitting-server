from django.http import JsonResponse
from django.shortcuts import render
import base64

def generate(request):
    if request.method == "POST":
        # Assuming 'image.jpg' is in the same directory as this script
        image_path = '/d:/Capstone/virtual-fitting-server/inference/image.jpg'
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return JsonResponse({
                'message': 'Inference Endpoint',
                'data': "This is the Inference Endpoint",
                'image': encoded_string
            }, status=200)
        except FileNotFoundError:
            return JsonResponse({
                'message': 'File not found',
                'data': "image.jpg not found",
            }, status=404)
    else:
        return JsonResponse({
            'message': 'Only POST is allowed',
            'data': "Only POST is allowed",
        }, status=405)