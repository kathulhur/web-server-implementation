import json
from . import abstraction
from rest_framework.exceptions import APIException
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from .serializers import InferenceSerializer
from . import common


class InformationView(APIView):
    def get(self, request):
        return HttpResponse(
            json.dumps(common.inference_metadata).encode(), 
            status=200, 
            content_type='application/json'
        )

class InferenceView(APIView):

    def post(self, request):
        serializer = InferenceSerializer(data=request.data)
        if not common.inferencing_module or not common.builder_class:
            raise APIException('Inferencing module was not properly setup')

        if serializer.is_valid():

            modelBuilder: abstraction.ModelBuilder = common.builder_class()

            input_files = serializer.validated_data.get('input_files')
            model_artifacts = serializer.validated_data.get('model_artifacts')

            model: abstraction.Model = modelBuilder.build(model_artifacts)
            result = model.infer(input_files)


            return HttpResponse(
                result.get('data').open('rb').read(), 
                content_type=result['type'], 
                status=200
            )
        
        return Response(serializer.errors, content_type='application/json', status=status.HTTP_400_BAD_REQUEST)
