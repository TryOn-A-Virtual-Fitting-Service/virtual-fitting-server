import subprocess
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import runpy
import shutil
import time
from datetime import datetime

@csrf_exempt
def generate(request):
    if request.method != "POST":
        return JsonResponse({
            'message': 'Only POST is allowed',
            'data': "Only POST is allowed",
        }, status=405)
    
    if 'clothing' not in request.FILES or 'model' not in request.FILES:
        return JsonResponse({
            'message': 'Missing files',
            'data': "Both clothing and model files are required",
        }, status=400)
    
    clothing = request.FILES['clothing']
    model = request.FILES['model']
    
    time.sleep(10)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # Set root_dir to 'virtual-fitting-server'
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f'result_{timestamp}.png'
    fitting_results_dir = os.path.join(root_dir, 'results')
    result_path = os.path.join(fitting_results_dir, filename)

    # fitting_results_dir에 이미 존재하는 파일 삭제
    if os.path.exists(fitting_results_dir):
        for file in os.listdir(fitting_results_dir):
            file_path = os.path.join(fitting_results_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    # 기존 이미지 파일 복사
    source_image_path = os.path.join(current_dir, 'image.png')
    shutil.copyfile(source_image_path, result_path)
    
    return JsonResponse({
        'message': 'Success',
        'data': f'{filename}',
    }, status=200)