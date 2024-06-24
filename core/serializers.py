from rest_framework import serializers
from django.core.files.uploadedfile import TemporaryUploadedFile, InMemoryUploadedFile
from . import common


class InferenceSerializer(serializers.Serializer):
    input_files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )
    model_artifacts = serializers.ListField(
        child=serializers.FileField(),
        required=False,
        default=list
    )

    def validate_input_files(self, value):
        if len(value) != len(common.inference_metadata['input_files']):
            raise serializers.ValidationError(f'There must be {len(common.inference_metadata["input_files"])} input files attached')
        return value

    def validate_model_artifacts(self, value: list[TemporaryUploadedFile | InMemoryUploadedFile]):
        if len(value) != len(common.inference_metadata['model_artifacts']):
            raise serializers.ValidationError(f'There must be {len(common.inference_metadata["model_artifacts"])} model artifacts attached')


        for i in range(len(value)):
            file = value[i]
            if isinstance(file, TemporaryUploadedFile):
                file_extension = '.' + file.name.split('.')[-1]
                if file_extension not in common.inference_metadata['model_artifacts'][i]:
                    raise serializers.ValidationError(f'The file extension of a model artifact is invalid')
                
            elif isinstance(file, InMemoryUploadedFile):
                file_extension = '.' + file.name.split('.')[-1]
                if file_extension not in common.inference_metadata['model_artifacts'][i]:
                    raise serializers.ValidationError(f'The file extension of a model artifact is invalid')
                

        return value