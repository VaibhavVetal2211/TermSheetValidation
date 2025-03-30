import unittest
from app import create_app

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_health_check(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertIn('healthy', response.get_json()['status'])

    def test_process_document(self):
        with open('tests/test_file.pdf', 'rb') as file:
            response = self.client.post('/process', data={'file': file})
            self.assertEqual(response.status_code, 200)
            self.assertIn('extracted_terms', response.get_json())

    def test_process_document_no_file(self):
        response = self.client.post('/process')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No file provided', response.get_json()['error'])

    def test_process_document_invalid_file_type(self):
        response = self.client.post('/process', data={'file': (BytesIO(b"test"), 'test.txt')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('File type not allowed', response.get_json()['error'])

if __name__ == '__main__':
    unittest.main()