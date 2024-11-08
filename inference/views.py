from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import runpy
import shutil
from datetime import datetime

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
        root_dir = os.path.dirname(current_dir)  # Set root_dir to 'virtual-fitting-server'
        
        temp_storage_dir = os.path.join(root_dir, 'temp_storage')
        clothing_path = os.path.join(temp_storage_dir, 'clothing.jpg')
        model_path = os.path.join(temp_storage_dir, 'model.jpg')
        output_path = os.path.join(root_dir, 'OOTDiffusion', 'run', 'images_output', 'out_hd_0.png')
        fitting_results_dir = os.path.join(root_dir, 'fitting_results')
        run_ootd_path = os.path.join(root_dir, 'OOTDiffusion', 'run', 'run_ootd.py')
        
        if not os.path.exists(fitting_results_dir):
            try:
                os.makedirs(fitting_results_dir)
            except OSError as e:
                return JsonResponse({
                    'message': 'Error creating fitting results directory',
                    'data': str(e),
                }, status=500)
        
        try:
            # 업로드된 옷과 모델 이미지를 저장
            with open(clothing_path, 'wb+') as destination:
                for chunk in clothing.chunks():
                    destination.write(chunk)
            with open(model_path, 'wb+') as destination:
                for chunk in model.chunks():
                    destination.write(chunk)
        except Exception as e:
            return JsonResponse({
                'message': 'Error saving files',
                'data': str(e),
            }, status=500)
        
        try:
            runpy.run_path(run_ootd_path, run_name='__main__', init_globals={
                'model_path': model_path,
                'cloth_path': clothing_path,
                'scale': 1.0,
                'sample': 1
            })
        except Exception as e:
            return JsonResponse({
                'message': 'Error running OOTD script',
                'data': str(e),
            }, status=500)
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            result_filename = f'result_{timestamp}.png'
            result_path = os.path.join(fitting_results_dir, result_filename)
            shutil.move(output_path, result_path)
        except Exception as e:
            return JsonResponse({
                'message': 'Error moving result file',
                'data': str(e),
            }, status=500)
        
        return JsonResponse({
            'message': 'Inference successful',
            'data': result_filename,
        }, status=200)
    else:
        return JsonResponse({
            'message': 'Only POST is allowed',
            'data': "Only POST is allowed",
        }, status=405)
    

@csrf_exempt
def retrieve(request):
    if request.method == "GET":
        if 'filename' not in request.GET:
            return JsonResponse({
                'message': 'Missing filename',
                'data': "Filename is required",
            }, status=400)
        
        filename = request.GET['filename']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fitting_results_dir = os.path.join(root_dir, 'fitting_results')
        result_path = os.path.join(fitting_results_dir, filename)
        
        if not os.path.exists(result_path):
            return JsonResponse({
                'message': 'File not found',
                'data': "File not found",
            }, status=404)
        
        with open(result_path, 'rb') as f:
            file_data = f.read()
            response = HttpResponse(file_data, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
    else:
        return JsonResponse({
            'message': 'Only GET is allowed',
            'data': "Only GET is allowed",
        }, status=405)
    