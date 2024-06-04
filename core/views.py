import importlib, pathlib
from . import abstraction
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.exceptions import APIException
from rest_framework.views import APIView
from django.http import HttpResponse
# Create your views here.

inferencing_module = None
builder_class = None

MODULE_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / 'output'

if not inferencing_module:
    print('====== inferencing module imported ========')
    inferencing_module = importlib.import_module('inference.implementation')

if not builder_class:
    if not inferencing_module:
        raise ImportError('Inferencing module is missing')
    
    print('====== builder class imported ========')
    builder_class = getattr(inferencing_module, 'builder_class')


class InferencingView(APIView):



    def get(self, request):
        if not inferencing_module or not builder_class:
            raise APIException('Inferencing module was not properly setup')

        model_artifacts_file_paths = [MODULE_DIR / 'yolov8n.pt']
        modelBuilder: abstraction.ModelBuilder = builder_class(model_artifacts_file_paths)
        model: abstraction.Model = modelBuilder.build_model()
        sample_image_inputs_file_path = [ MODULE_DIR / 'image.png']
        result = model.infer(sample_image_inputs_file_path)

        data = None
        print('result path', result['data'])
        with result['data'].open('rb') as f:
            data = f.read()

        return HttpResponse(data, content_type=result['type'], status=200)
