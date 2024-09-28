from django.http import JsonResponse
from django.shortcuts import render
import base64
from django.views.decorators.csrf import csrf_exempt
import os

@csrf_exempt
def generate(request):
    if request.method == "POST":
        if 'clothing' not in request.FILES or 'model' not in request.FILES:
            return JsonResponse({
                'message': 'Missing files',
                'data': "Both clothing and model files are required",
            }, status=400)

        clothing = request.FILES['clothing']
        model = request.FILES['model']

        current_dir = os.path.dirname(os.path.abspath(__file__))
        clothing_path = os.path.join(current_dir, 'clothing.jpg')
        model_path = os.path.join(current_dir, 'model.jpg')

        try:
            with open(clothing_path, 'wb+') as destination:
                for chunk in clothing.chunks():
                    destination.write(chunk)

            with open(model_path, 'wb+') as destination:
                for chunk in model.chunks():
                    destination.write(chunk)

            return JsonResponse({
                'message': 'Files received and saved successfully',
                'data': "Files have been saved",
            }, status=200)
        except Exception as e:
            return JsonResponse({
                'message': 'Error saving files',
                'data': str(e),
            }, status=500)
    else:
        return JsonResponse({
            'message': 'Only POST is allowed',
            'data': "Only POST is allowed",
        }, status=405)
