from django.test import TestCase
from django.contrib.auth.models import User
from diagnosis.models import DiagnosisResult

class DiagnosisResultModelTest(TestCase):
    def setUp(self):
        # Create a test user
        self.user = User.objects.create_user(username='testuser', password='pass123')

    def test_diagnosis_creation(self):
        # Create a DiagnosisResult
        diag = DiagnosisResult.objects.create(
            user=self.user,
            label='strep',
            probability=0.85
        )
        # Ensure the object was saved
        self.assertIsNotNone(diag.id)
        self.assertEqual(diag.label, 'strep')
        self.assertAlmostEqual(diag.probability, 0.85, places=2)

    def test_diagnosis_string_representation(self):
        diag = DiagnosisResult.objects.create(
            user=self.user,
            label='healthy',
            probability=0.3
        )
        diag_str = str(diag)
        self.assertIn(self.user.username, diag_str)
        self.assertIn('healthy', diag_str)
