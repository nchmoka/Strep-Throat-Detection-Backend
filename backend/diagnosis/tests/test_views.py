# diagnosis/tests/test_views.py

import os
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from diagnosis.models import DiagnosisResult

class DiagnosisViewTests(TestCase):
    def setUp(self):
        self.client = Client()

        # URLs must match your urls.py definitions
        self.register_url = reverse('register')            # e.g. path('register/', register_user, name='register')
        self.login_url = reverse('login')                  # e.g. path('login/', login_user, name='login')
        self.analyze_url = reverse('analyze')              # e.g. path('analyze/', analyze_image, name='analyze')
        self.history_url = reverse('analysis_history')     # e.g. path('analysis/history/', analysis_history, name='analysis_history')

    #
    # Registration Tests
    #
    def test_register_user_success(self):
        response = self.client.post(self.register_url, {
            'username': 'testuser',
            'password': 'pass123'
        })
        self.assertEqual(response.status_code, 201)
        self.assertIn('User registered successfully', response.content.decode())

        self.assertTrue(User.objects.filter(username='testuser').exists())

    def test_register_user_duplicate(self):
        User.objects.create_user(username='testuser', password='pass123')
        response = self.client.post(self.register_url, {
            'username': 'testuser',
            'password': 'pass123'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Username already exists', response.content.decode())

    #
    # Login Tests
    #
    def test_login_success(self):
        User.objects.create_user(username='testuser', password='pass123')
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'pass123'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Login successful', response.content.decode())

        # Check session-based authentication
        self.assertIn('_auth_user_id', self.client.session)

    def test_login_failure(self):
        response = self.client.post(self.login_url, {
            'username': 'nonexisting',
            'password': 'wrongpass'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid credentials', response.content.decode())

    #
    # Analyze Tests
    #
    def test_analyze_not_authenticated(self):
        # Expect 401 if user is not logged in
        response = self.client.post(self.analyze_url)
        self.assertEqual(response.status_code, 401)
        self.assertIn('Authentication required', response.content.decode())

    def test_analyze_success(self):
        # Create & log in user
        user = User.objects.create_user(username='testuser', password='pass123')
        self.client.login(username='testuser', password='pass123')

        # Use a real image from testdata
        testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')
        image_path = os.path.join(testdata_dir, 'throat.jpg')  # must be a valid JPEG

        with open(image_path, 'rb') as f:
            mock_image = SimpleUploadedFile(
                name='throat.jpg',
                content=f.read(),
                content_type='image/jpeg'
            )

        response = self.client.post(self.analyze_url, {'image': mock_image})
        self.assertEqual(response.status_code, 200, msg=response.content.decode())

        data = response.json()
        self.assertIn('prediction', data)
        self.assertIn('probability', data)

        # Verify a DiagnosisResult was created
        diag = DiagnosisResult.objects.filter(user=user).first()
        self.assertIsNotNone(diag)
        self.assertIn(diag.label, ['strep', 'healthy'])

    #
    # History Tests
    #
    def test_analysis_history_not_authenticated(self):
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, 401)
        self.assertIn('Authentication required', response.content.decode())

    def test_analysis_history_success(self):
        # Create & log in user
        user = User.objects.create_user(username='historyuser', password='pass123')
        self.client.login(username='historyuser', password='pass123')

        # Create some test records
        DiagnosisResult.objects.create(user=user, label='strep', probability=0.78)
        DiagnosisResult.objects.create(user=user, label='healthy', probability=0.22)

        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)

        labels = [item['label'] for item in data]
        self.assertIn('strep', labels)
        self.assertIn('healthy', labels)


def test_register_empty_username(self):
    """
    Attempt to register with an empty username.
    Expected: HTTP 400 and an error message.
    """
    response = self.client.post(self.register_url, {
        'username': '',
        'password': 'pass123'
    })
    self.assertEqual(response.status_code, 400)
    self.assertIn('Username and password required', response.content.decode())

def test_register_empty_password(self):
    """
    Attempt to register with an empty password.
    Expected: HTTP 400 and an error message.
    """
    response = self.client.post(self.register_url, {
        'username': 'noPasswordUser',
        'password': ''
    })
    self.assertEqual(response.status_code, 400)
    self.assertIn('Username and password required', response.content.decode())

def test_register_missing_fields(self):
    """
    Attempt to register with no form data at all.
    Expected: HTTP 400 and a relevant error or fallback message.
    """
    response = self.client.post(self.register_url, {})
    self.assertEqual(response.status_code, 400)
    self.assertIn('Username and password required', response.content.decode())
