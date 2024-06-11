from rest_framework.test import APITestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
import pathlib
# Create your tests here.

MODULE_PATH = pathlib.Path(__file__).parent
image_path = MODULE_PATH / 'assets' / 'image.png'
yolov8_path = MODULE_PATH / 'assets' / 'yolov8n.pt' 

class InferenceEndpointTest(APITestCase):

    def test_inference(self):
        with image_path.open('rb') as f:
            file_1 = SimpleUploadedFile(
                'file1.png',
                f.read(),
        )
            
        with yolov8_path.open('rb') as f:
            file_2 = SimpleUploadedFile(
                'file2.pt',
                f.read(),
            )

        response = self.client.post('/inference/', {
            'input_files': [file_1],
            'model_artifacts': [file_2]
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.headers.get('Content-Type'), 'image/png')

    def test_multiple_inference(self):

        with image_path.open('rb') as f:
            file_1 = SimpleUploadedFile(
                'file1.png',
                f.read(),
        )
            
        with yolov8_path.open('rb') as f:
            file_2 = SimpleUploadedFile(
                'file2.pt',
                f.read(),
            )

        response = self.client.post('/inference/', {
            'input_files': [file_1],
            'model_artifacts': [file_2]
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.headers.get('Content-Type'), 'image/png')
        
        with image_path.open('rb') as f:
            file_1 = SimpleUploadedFile(
                'file1.png',
                f.read(),
        )
            
        with yolov8_path.open('rb') as f:
            file_2 = SimpleUploadedFile(
                'file2.pt',
                f.read(),
            )

        response = self.client.post('/inference/', {
            'input_files': [file_1],
            'model_artifacts': [file_2]
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.headers.get('Content-Type'), 'image/png')


    def test_input_files_length_does_not_match(self):
            with image_path.open('rb') as f:
                file_1 = SimpleUploadedFile(
                    'file1.png',
                    f.read(),
            )
                
            with image_path.open('rb') as f:
                file_2 = SimpleUploadedFile(
                    'file2.png',
                    f.read(),
            )
                
            with yolov8_path.open('rb') as f:
                file_3 = SimpleUploadedFile(
                    'file3.pt',
                    f.read(),
                )

            response = self.client.post('/inference/', {
                'input_files': [file_1, file_2],
                'model_artifacts': [file_3]
            })
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
            self.assertIn('input_files', response.data)

    def test_model_artifacts_length_does_not_match(self):
        with image_path.open('rb') as f:
            file_1 = SimpleUploadedFile(
                'file1.png',
                f.read(),
            )
            
        with yolov8_path.open('rb') as f:
            file_2 = SimpleUploadedFile(
                'file2.pt',
                f.read(),
            )

        with yolov8_path.open('rb') as f:
            file_3 = SimpleUploadedFile(
                'file3.pt',
                f.read(),
            )

        response = self.client.post('/inference/', {
            'input_files': [file_1],
            'model_artifacts': [file_2, file_3]
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('model_artifacts', response.data)

    def test_model_artifacts_with_invalid_file_extensions_not_allowed(self):
        with image_path.open('rb') as f:
            file_1 = SimpleUploadedFile(
                'file1.png',
                f.read(),
            )
            
        with yolov8_path.open('rb') as f:
            file_2 = SimpleUploadedFile(
                'file2.pta',
                f.read(),
            )


        response = self.client.post('/inference/', {
            'input_files': [file_1],
            'model_artifacts': [file_2]
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('model_artifacts', response.data)




class InformationEndpointTest(APITestCase):
     

    def test_information_endpoint_request(self):
        response = self.client.get('/info/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_information_endpoint_contains_expected_fields(self):
        response = self.client.get('/info/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.headers.get('Content-Type'), 'application/json')
        response_data = response.json()
        self.assertIn('input_files', response_data)
        self.assertIn('model_artifacts', response_data)
        
     