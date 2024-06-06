import pathlib
from typing import TypedDict



InferenceResult = TypedDict('InferenceResult', {
    'data': pathlib.Path,
    'type': str,
    'info': dict
})
     
class Model:
    """
      An object that can infer or predict
      Contains every knowledge about performing the inference given an input
    """
    def infer(self, input_file_paths: list[pathlib.Path]) -> InferenceResult:
        raise NotImplementedError()
    

class ModelBuilder:
    """
        An object that can build the inference model given a list of model artifacts
        contains every logic that it needs to build the model
    """
    def build(self, model_file_paths: list[pathlib.Path]) -> Model:
        raise NotImplementedError()

   
