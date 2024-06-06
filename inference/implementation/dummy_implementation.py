import pathlib
from typing import TypedDict
from .abstraction import Model, ModelBuilder, InferenceResult, FilePath

module_path = pathlib.Path(__file__).parent
print(module_path)



class DummyModel(Model):
    """
      An object that can infer or predict
      Contains every knowledge about performing the inference given an input
    """
    def infer(self, input_file_paths: list[FilePath]) -> InferenceResult:
        return {
            'data': module_path / 'assets' / 'image.png',
            'type': 'image/png',
            'info': {}
        }
    

class DummyModelBuilder(ModelBuilder):
    """
        An object that can build the inference model given a list of model artifacts
        contains every logic that it needs to build the model
    """
    def build(self, model_file_paths: list[FilePath]) -> DummyModel:
        return DummyModel()

   



builder_class = DummyModelBuilder
inference_metadata = {
    'input_files': [ ['image'] ],
    'model_artifacts': [ ['.pt'] ],
}