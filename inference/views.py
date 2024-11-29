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
import onnxruntime as ort
# from OOTDiffusion.run.run_ootd import run_ootd

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

    print(f"Given protocol is : {request.scheme}")
    clothing_url = request.POST.get('clothing')
    model_url = request.POST.get('model')

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

    # 환경 설정
    os.environ["U2NET_HOME"] = os.path.join(BASE_DIR, "static", "u2net")  # U2Net 모델 경로

    headers = {'User-Agent': 'Mozilla/5.0'}

    # 의류 이미지 다운로드
    try:
        clothing_response = requests.get(clothing_url, headers=headers)
        clothing_response.raise_for_status()
        with open(clothing_path, 'wb') as destination:
            destination.write(clothing_response.content)
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'message': 'Error downloading clothing file',
            'data': str(e),
        }, status=500)

    # 모델 이미지 다운로드
    try:
        model_response = requests.get(model_url, headers=headers)
        model_response.raise_for_status()
        with open(model_path, 'wb') as destination:
            destination.write(model_response.content)
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'message': 'Error downloading model file',
            'data': str(e),
        }, status=500)

    # # 배경 제거 처리
    # try:
    #     clothing_image = Image.open(clothing)
    #     model_image = Image.open(model)

    #     # rembg를 사용하여 배경 제거
    #     clothing_image = remove(clothing_image)  # 세션이 필요 없음
    #     model_image = remove(model_image)

    #     clothing_image = clothing_image.convert("RGB")
    #     model_image = model_image.convert("RGB")

    #     # 결과를 저장할 메모리 버퍼
    #     clothing = BytesIO()
    #     model = BytesIO()

    #     # 배경에 흰색 추가
    #     background_color = (255, 255, 255)  # 흰색
    #     background_clothing = Image.new("RGB", clothing_image.size, background_color)
    #     background_model = Image.new("RGB", model_image.size, background_color)

    #     background_clothing.paste(clothing_image, mask=clothing_image.split()[3])
    #     background_model.paste(model_image, mask=model_image.split()[3])

    #     # JPEG 형식으로 저장
    #     background_clothing.save(clothing_path, format='JPEG')
    #     background_model.save(model_path, format='JPEG')

    # except Exception as e:
    #     return JsonResponse({
    #         'message': 'Error processing images',
    #         'data': str(e),
    #     }, status=500)

    # background_clothing.paste(clothing_image, mask=clothing_image.split()[3])
    # background_model.paste(model_image, mask=model_image.split()[3])

    # background_clothing.save(clothing_path, format='JPEG')
    # background_model.save(model_path, format='JPEG')

    # ############################################ TEST ############################################    
    # return JsonResponse({
    #     'message': 'Test successful',
    # }, status=200)
    # ##############################################################################################



    # clothing = Image.open(clothing)
    # model = Image.open(model)

    # clothing = remove(clothing)
    # model = remove(model)

    # clothing_data = clothing.read()
    # model_data = model.read()
    
    from run.run_ootd import run_ootd

    image = run_ootd(model_path, clothing_path)
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

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
