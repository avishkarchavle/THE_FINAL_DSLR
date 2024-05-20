import unittest
from app import app
class FlaskTestCase(unittest.TestCase):
       def setUp(self):
           self.app = app.test_client()
           self.app.testing = True

       def test_predict_endpoint(self):
           with open('00336.mp4', 'rb') as video:
               response = self.app.post('/predict', data={'video': video})
               self.assertEqual(response.status_code, 200)
               self.assertIn('prediction', response.json)

if __name__ == '__main__':
       unittest.main()
   
