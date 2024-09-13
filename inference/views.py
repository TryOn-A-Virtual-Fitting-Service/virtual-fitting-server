from django.http import JsonResponse
from django.shortcuts import render

def generate(request):
    if request.method == "GET":
        return JsonResponse({
            'message': 'Inference Endpoint',
            'data': "This is the Inference Endpoint",
        }, status=200)
    else:
        return JsonResponse({
            'message': 'Only GET is allowed',
            'data': "Only GET is allowed",
        }, status=405)