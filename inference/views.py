import subprocess
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import runpy
import shutil
from datetime import datetime
from rembg import remove
# from OOTDiffusion.run.run_ootd import run_ootd

@csrf_exempt
def generate(request):
    if request.method != "POST":
        return JsonResponse({
            'message': 'Only POST is allowed',
            'data': "Only POST is allowed",
        }, status=405)
    
    if 'clothing' not in request.FILES:
        return JsonResponse({
            'message': 'Missing clothing file',
            'data': "Clothing file is required",
        }, status=400)
    
    if 'model' not in request.FILES:
        return JsonResponse({
            'message': 'Missing model file',
            'data': "Model file is required",
        }, status=400)
    
    clothing = request.FILES['clothing']
    model = request.FILES['model']

    # save the uploaded files to the temp_storage directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    temp_storage_dir = os.path.join(root_dir, 'temp_storage')

    clothing_path = os.path.join(temp_storage_dir, 'clothing.jpg')
    model_path = os.path.join(temp_storage_dir, 'model.jpg')

    # clothing_data = clothing.read()
    # model_data = model.read()

    # clothing = remove(clothing_data)
    # model = remove(model_data)

    with open(clothing_path, 'wb+') as destination:
        for chunk in clothing.chunks():
            destination.write(chunk)
    
    with open(model_path, 'wb+') as destination:
        for chunk in model.chunks():
            destination.write(chunk)
    
    from run.run_ootd import run_ootd
    image = run_ootd(model_path, clothing_path)
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        result_filename = f'result_{timestamp}.png'
        image.save(f'./results/result_{timestamp}.png')
    except Exception as e:
        return JsonResponse({
            'message': 'Error saving result file',
            'data': str(e),
        }, status=500)
    
    return JsonResponse({
        'message': 'Inference successful',
        'data': result_filename,
    }, status=200)
