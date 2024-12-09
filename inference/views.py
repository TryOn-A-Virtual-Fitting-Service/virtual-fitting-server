import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import requests
import shutil
from datetime import datetime
from rembg import remove
from PIL import Image
import torch
from io import BytesIO
import json
import onnxruntime as ort
import gc
# from OOTDiffusion.run.run_ootd import run_ootd

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='fp16')

@csrf_exempt
def generate(request):
    if request.method != "POST":
        return JsonResponse({
            'message': 'Only POST is allowed',
            'data': "Only POST is allowed",
        }, status=405)
    
    print("-----------------------------------\n\n")
    print("Request headers: ")
    print(request.headers)
    # print("Request body: ")
    # print(request.body)
    # print("Request files: ")
    # print(request.FILES)

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Given protocol is : {request.scheme}")
    try:
        data = json.loads(request.body)
        clothing_url = data.get('clothing')
        model_url = data.get('model')
        category = data.get('category')
    except json.JSONDecodeError:
        return JsonResponse({
            'message': 'Invalid JSON input',
            'data': "Invalid JSON input",
        }, status=400)

    if not clothing_url:
        return JsonResponse({
            'message': 'Missing clothing URL',
            'data': "Clothing URL is required",
        }, status=400)
    
    if not model_url:
        return JsonResponse({
            'message': 'Missing model URL',
            'data': "Model URL is required",
        }, status=400)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    temp_storage_dir = os.path.join(root_dir, 'temp_storage')
    clothing_path = os.path.join(temp_storage_dir, 'clothing.jpg')
    model_path = os.path.join(temp_storage_dir, 'model.jpg')

    headers = {'User-Agent': 'Mozilla/5.0'}

    def download_image(url, path, headers={'User-Agent': 'Mozilla/5.0'}):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(path, 'wb') as destination:
                destination.write(response.content)
            print(f"URL: {url}")
            print(f"  Image downloaded successfully: {path}")
        except requests.exceptions.RequestException as e:
            return str(e)
        return None

    from PIL import Image, ImageOps

    def resize_and_convert_image(image_path, max_width, max_height):
        print(f"  Processing image: {image_path}")
        try:
            with Image.open(image_path) as img:
                # 이미지의 현재 크기 가져오기
                width, height = img.size

                # 축소 비율 계산 (비율을 유지하면서 최대 크기 안에 들어오도록)
                ratio = min(max_width / width, max_height / height, 1)

                # 새로운 크기 계산
                new_size = (int(width * ratio), int(height * ratio))

                # 이미지를 축소해야 하는 경우에만 리사이즈 수행
                # if ratio < 1: # 잠시 비활성화
                #     img = img.resize(new_size, Image.Resampling.LANCZOS)
                #     print(f"  Image resized to {new_size}")
                # else:
                #     print("  Image size is within the maximum bounds. No resizing needed.")

                # 알파 채널 처리
                if img.mode == 'RGBA':
                    alpha = img.split()[3]
                    img = img.convert('RGB')
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=alpha)
                    img = background
                else:
                    img = img.convert('RGB')  # RGB 모드로 변환

                # 이미지를 JPEG로 저장
                img.save(image_path, 'JPEG')
                print(f"  Image processed and saved as JPEG: {image_path}")
        except Exception as e:
            return str(e)
        return None

    import concurrent.futures

    # 다운로드 단계 시작 시간 측정
    start_time_download = time.time()
    
    print("Downloading images...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_clothing = executor.submit(download_image, clothing_url, clothing_path, headers)
        future_model = executor.submit(download_image, model_url, model_path, headers)

        clothing_error = future_clothing.result()
        model_error = future_model.result()
    
    # 다운로드 단계 종료 시간 및 실행 시간 계산
    end_time_download = time.time()
    duration_download = (end_time_download - start_time_download) * 1000  # 밀리초 단위
    print(f"### Download step : {duration_download:.2f} ms ###\n")
    
    if clothing_error:
        return JsonResponse({
            'message': 'Error downloading clothing file',
            'data': clothing_error,
        }, status=500)

    if model_error:
        return JsonResponse({
            'message': 'Error downloading model file',
            'data': model_error,
        }, status=500)

    # 리사이징 및 변환 단계 시작 시간 측정
    start_time_resize = time.time()

    # 이미지 리사이징 및 저장을 병렬로 처리
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_clothing = executor.submit(resize_and_convert_image, clothing_path, 1280, 960)
        future_model = executor.submit(resize_and_convert_image, model_path, 1280, 960)

        clothing_resize_error = future_clothing.result()
        model_resize_error = future_model.result()

    # 리사이징 및 변환 단계 종료 시간 및 실행 시간 계산
    end_time_resize = time.time()
    duration_resize = (end_time_resize - start_time_resize) * 1000  # 밀리초 단위
    print(f"### Resize and conversion : {duration_resize:.2f} ms ###\n")

    if clothing_resize_error:
        return JsonResponse({
            'message': 'Error resizing clothing image',
            'data': clothing_resize_error,
        }, status=500)

    if model_resize_error:
        return JsonResponse({
            'message': 'Error resizing model image',
            'data': model_resize_error,
        }, status=500)

    # 이미지 저장 단계 시작 시간 측정
    start_time_save = time.time()

    # 추가적인 저장 단계가 없다면, 이 부분은 생략 가능합니다.
    # 여기서는 예시로 빈 작업을 수행합니다.
    print("  Images are already saved during resizing step.")

    # 이미지 저장 단계 종료 시간 및 실행 시간 계산
    end_time_save = time.time()
    duration_save = (end_time_save - start_time_save) * 1000  # 밀리초 단위
    print(f"### Image Saving : {duration_save:.2f} ms ###\n")

    # ############################################ TEST ############################################    
    # return JsonResponse({
    #     'message': 'Test successful',
    # }, status=200)
    # ##############################################################################################

    from run.run_ootd import run_ootd
    if int(category) == 0: # Upper
        category = 0
        model_type = 'hd'
    elif int(category) == 1: # Lower
        category = 1
        model_type = 'dc'
    else:
        return JsonResponse({
            'message': 'Invalid category',
            'data': "Category must be 0 (upper) or 1 (lower)",
        }, status=400)
    try:
        image = run_ootd(model_path, clothing_path, accelerator, model_type=model_type, category=category, scale=2.0, step=40)
    except Exception as e:
        return JsonResponse({
            'message': 'Error running OOTD model',
            'data': str(e),
        }, status=500)
    

    # 모델 사이즈에 맞게 이미지 크기 조정
    with Image.open(model_path) as model_img:
        model_width, model_height = model_img.size
    image = image.resize((model_width, model_height), Image.Resampling.LANCZOS)

    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        result_filename = f'result_{timestamp}.png'
        image.save(f'./results/result_{timestamp}.png')
    except Exception as e:
        return JsonResponse({
            'message': 'Error saving result file',
            'data': str(e),
        }, status=500)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return JsonResponse({
        'message': 'Inference successful',
        'data': result_filename,
    }, status=200)
