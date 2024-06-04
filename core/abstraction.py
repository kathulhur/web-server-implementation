
class ModelBuilder:
    """
        An object that can build/product a model given a file
    """
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def build_model(self):
        raise NotImplementedError()

class Model:
    """
        An object that can infer/predict given an input file
    """
    def __init__(self, model):
        self.model = model

    def infer(self, mode, input, *args, **kwargs):
        raise NotImplementedError()


class Prediction:
    """
        An object that represents a model prediction/inference holding the prediction result
    """
    def __init__(self, data, type, info={}):
        self.data = data
        self.type = type 
        self.info = info

    def get_result(self):
        raise NotImplementedError()
    

class PredictionPipeline:
    """
        An object that encapsulating the knowledge of conducting
            the inferencing process and knows how to return the 
            correct HTTP response based from the output the model
    """

    def __init__(self, model):
        pass
    
    def get_response(self, input_file_path: str, *args, **kwargs):
        raise NotImplementedError()