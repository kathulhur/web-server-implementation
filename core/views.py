import json, pathlib, os
from . import abstraction
from rest_framework.exceptions import APIException
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from .serializers import InferenceSerializer
from . import common
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.core.files.storage import Storage, default_storage
from typing import Union, List

MODULE_DIR = pathlib.Path(__file__).parent
TEMP_DIR = MODULE_DIR / 'temp'
storage: Storage = default_storage

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
        if not common.inferencing_module or not common.model_builder_class:
            raise APIException('Inferencing module was not properly setup')
        model_artifacts_file_paths = None
        input_file_paths = None
        try:
            if serializer.is_valid(raise_exception=True):

                modelBuilder: abstraction.ModelBuilder = common.model_builder_class()

                input_files: List[Union[InMemoryUploadedFile, TemporaryUploadedFile]] = serializer.validated_data.get('input_files')
                model_artifacts: List[Union[InMemoryUploadedFile, TemporaryUploadedFile]] = serializer.validated_data.get('model_artifacts')

                model_artifacts_file_paths = []
                for artifact in model_artifacts:
                    if isinstance(artifact, InMemoryUploadedFile):
                        temp_path = TEMP_DIR / artifact.name
                        available_name = storage.get_available_name(temp_path)
                        storage.save(temp_path, artifact.file)
                        model_artifacts_file_paths.append(str(temp_path))

                    else:
                        model_artifacts_file_paths.append(artifact.temporary_file_path())
                
                input_file_paths = []
                for artifact in input_files:
                    if isinstance(artifact, InMemoryUploadedFile):
                        temp_path = TEMP_DIR / artifact.name
                        available_name = storage.get_available_name(temp_path)
                        storage.save(available_name, artifact.file)
                        input_file_paths.append(str(available_name))

                    else:
                        input_file_paths.append(artifact.temporary_file_path())
                
                model: abstraction.Model = modelBuilder.build(model_artifacts_file_paths)
                result = model.infer(input_file_paths)

                resultData = None
                with open(result['data'], 'rb') as f:
                    resultData = f.read()

                return HttpResponse(
                    resultData, 
                    content_type=result['type'],
                    status=200
                )
        except APIException as e:
            return Response(serializer.errors, content_type='application/json', status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            print(e)
            return Response({ 'message': 'Server Error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            pass
            # Find a way to clean up the file without causing os exception
            # if input_file_paths:
            #     for file_path in input_file_paths:
            #         os.remove(file_path)

            # if model_artifacts_file_paths:
            #     for file_path in model_artifacts_file_paths:
            #         os.remove(file_path)

