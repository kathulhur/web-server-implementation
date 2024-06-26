import pathlib, importlib
from . import abstraction

MODULE_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / 'output'
inferencing_module = None
model_builder_class: abstraction.ModelBuilder = None
inference_metadata: abstraction.InferenceMetadata = None

if not inferencing_module:
    inferencing_module = importlib.import_module('inference.implementation')

if not model_builder_class:
    if not inferencing_module:
        raise ImportError('Inferencing module is missing')
    
    model_builder_class = getattr(inferencing_module, 'model_builder_class')


if not inference_metadata:
    if not inferencing_module:
        raise ImportError('Inferencing module is missing')
    
    inference_metadata = getattr(inferencing_module, 'inference_metadata')

