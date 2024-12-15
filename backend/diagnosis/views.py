from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse, HttpResponseBadRequest
import os
import numpy as np
from django.conf import settings
from PIL import Image
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
from .models import DiagnosisResult, EducationalResource
import json

# Load model once
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'strep_throat_cnn.keras')
model = load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username and password:
            if User.objects.filter(username=username).exists():
                return JsonResponse({"error": "Username already exists"}, status=400)
            user = User.objects.create_user(username=username, password=password)
            return JsonResponse({"message": "User registered successfully"}, status=201)
        return HttpResponseBadRequest("Username and password required")
    return HttpResponseBadRequest("Invalid request method")

@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({"message": "Login successful"}, status=200)
        else:
            return JsonResponse({"error": "Invalid credentials"}, status=400)
    return HttpResponseBadRequest("Invalid request method")

def logout_user(request):
    if request.user.is_authenticated:
        logout(request)
        return JsonResponse({"message": "Logged out successfully"}, status=200)
    return JsonResponse({"error": "Not logged in"}, status=400)

@csrf_exempt
def analyze_image(request):
    if request.method == 'POST':
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        if 'image' not in request.FILES:
            return JsonResponse({"error": "No image provided"}, status=400)

        image_file = request.FILES['image']
        img = Image.open(image_file).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        label = "strep" if pred > 0.5 else "healthy"

        # Save to database
        diagnosis = DiagnosisResult.objects.create(
            user=request.user,
            label=label,
            probability=float(pred),
            image=image_file  # store original image
        )

        return JsonResponse({
            "prediction": label,
            "probability": float(pred),
            "id": diagnosis.id
        }, status=200)
    return HttpResponseBadRequest("Invalid request method")

def list_resources(request):
    """
    GET /resources/
    Returns a list of all educational resources.
    """
    if request.method == 'GET':
        resources = EducationalResource.objects.all().order_by('-created_at')
        data = []
        for r in resources:
            data.append({
                "id": r.id,
                "title": r.title,
                "content": r.content,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat()
            })
        return JsonResponse(data, safe=False)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def resource_detail(request, resource_id):
    """
    GET /resources/<id>/
    Returns details of a single educational resource by ID.
    """
    if request.method == 'GET':
        try:
            r = EducationalResource.objects.get(pk=resource_id)
            data = {
                "id": r.id,
                "title": r.title,
                "content": r.content,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat()
            }
            return JsonResponse(data, status=200)
        except EducationalResource.DoesNotExist:
            return HttpResponseNotFound("Resource not found")
    return JsonResponse({"error": "Method not allowed"}, status=405)