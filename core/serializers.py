from rest_framework import serializers
from . import abstraction
from . import common


class InferenceSerializer(serializers.Serializer):
    input_files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )
    model_artifacts = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )

    def validate_input_files(self, value):
        if len(value) != len(common.inference_metadata['input_files']):
            raise serializers.ValidationError(f'There must be {len(common.inference_metadata['input_files'])} input files attached')
        return value

    def validate_model_artifacts(self, value):
        if len(value) != len(common.inference_metadata['model_artifacts']):
            raise serializers.ValidationError(f'There must be {len(common.inference_metadata['model_artifacts'])} model artifacts attached')
        return value