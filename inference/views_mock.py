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
    
    # if 'clothing' not in request.FILES or 'model' not in request.FILES:
    #     return JsonResponse({
    #         'message': 'Missing files',
    #         'data': "Both clothing and model files are required",
    #     }, status=400)
    
    # clothing = request.FILES['clothing']
    # model = request.FILES['model']
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # Set root_dir to 'virtual-fitting-server'
    
    # mock서버 이므로 time.sleep(10)을 한 후 현재 타임스탬프로 파일을 생성하여 반환

    time.sleep(10)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    filename = f'result_{timestamp}.png'
    fitting_results_dir = os.path.join(root_dir, 'results')
    result_path = os.path.join(fitting_results_dir, filename)

    # result_path에 이미 존재하는 파일 삭제
    if os.path.exists(result_path):
        if os.path.isdir(result_path):
            shutil.rmtree(result_path)
        else:
            os.remove(result_path)
    
    # 기존 이미지 파일 복사
    source_image_path = os.path.join(current_dir, 'image.png')
    shutil.copyfile(source_image_path, result_path)
    
    return JsonResponse({
        'message': 'Success',
        'data': f'{filename}',
    }, status=200)