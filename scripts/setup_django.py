#!/usr/bin/env python3
"""
Django Setup Script for Robo-Poet Web Interface
Initializes Django project with all necessary configurations
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_django_project():
    """Setup Django project structure"""

    print("[DJANGO] Setting up Robo-Poet Web Interface")
    print("=" * 60)

    # Install Django and dependencies
    print("\n[1/5] Installing Django dependencies...")
    requirements = [
        "Django==5.0.6",
        "djangorestframework==3.15.1",
        "django-cors-headers==4.3.1",
        "django-extensions==3.2.3",
        "python-decouple==3.8",
        "whitenoise==6.6.0",
        "gunicorn==21.2.0"
    ]

    for package in requirements:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package],
                      capture_output=True)

    # Create Django project
    print("\n[2/5] Creating Django project structure...")

    # Check if robo_poet_web already exists
    web_dir = Path("robo_poet_web")
    if not web_dir.exists():
        subprocess.run(["django-admin", "startproject", "robo_poet_web", "."])
        print("[OK] Django project created")
    else:
        print("[INFO] Django project already exists")

    # Create apps
    print("\n[3/5] Creating Django apps...")
    apps = ["training", "generation", "api", "dashboard"]

    for app in apps:
        app_path = Path(app)
        if not app_path.exists():
            subprocess.run(["python", "manage.py", "startapp", app])
            print(f"[OK] Created app: {app}")
        else:
            print(f"[INFO] App {app} already exists")

    # Create directories
    print("\n[4/5] Creating project directories...")
    directories = [
        "static",
        "static/css",
        "static/js",
        "static/img",
        "media",
        "media/checkpoints",
        "media/datasets",
        "templates",
        "templates/base",
        "templates/training",
        "templates/generation",
        "templates/dashboard"
    ]

    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
    print("[OK] Directories created")

    # Create initial templates
    print("\n[5/5] Creating initial templates...")

    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Robo-Poet AI{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    {% block extra_css %}{% endblock %}
</head>
<body class="bg-gray-900 text-white">
    <nav class="bg-gray-800 p-4">
        <div class="container mx-auto flex justify-between items-center">
            <div class="flex items-center space-x-2">
                <i class="fas fa-robot text-2xl text-purple-500"></i>
                <h1 class="text-2xl font-bold">Robo-Poet AI</h1>
            </div>
            <ul class="flex space-x-6">
                <li><a href="{% url 'dashboard' %}" class="hover:text-purple-400">Dashboard</a></li>
                <li><a href="{% url 'training' %}" class="hover:text-purple-400">Training</a></li>
                <li><a href="{% url 'generation' %}" class="hover:text-purple-400">Generate</a></li>
                <li><a href="{% url 'models' %}" class="hover:text-purple-400">Models</a></li>
            </ul>
        </div>
    </nav>

    <main class="container mx-auto p-6">
        {% block content %}{% endblock %}
    </main>

    <footer class="bg-gray-800 p-4 mt-12">
        <div class="container mx-auto text-center">
            <p>&copy; 2025 Robo-Poet AI | GPU: {{ gpu_status|default:"Checking..." }}</p>
        </div>
    </footer>

    {% block extra_js %}{% endblock %}
</body>
</html>'''

    with open("templates/base/base.html", "w") as f:
        f.write(base_template)

    # Dashboard template
    dashboard_template = '''{% extends "base/base.html" %}

{% block title %}Dashboard - Robo-Poet AI{% endblock %}

{% block content %}
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    <!-- GPU Status Card -->
    <div class="bg-gray-800 rounded-lg p-6">
        <h2 class="text-xl font-bold mb-4 flex items-center">
            <i class="fas fa-microchip mr-2 text-green-500"></i>
            GPU Status
        </h2>
        <div class="space-y-2">
            <p>Device: <span class="text-green-400">{{ gpu_name }}</span></p>
            <p>Memory: <span id="gpu-memory">{{ gpu_memory }}</span></p>
            <p>Temperature: <span id="gpu-temp">{{ gpu_temp }}</span></p>
            <p>Utilization: <span id="gpu-util">{{ gpu_util }}</span></p>
        </div>
    </div>

    <!-- Active Trainings Card -->
    <div class="bg-gray-800 rounded-lg p-6">
        <h2 class="text-xl font-bold mb-4 flex items-center">
            <i class="fas fa-brain mr-2 text-purple-500"></i>
            Active Trainings
        </h2>
        <div id="active-trainings">
            {% for training in active_trainings %}
            <div class="mb-3 p-3 bg-gray-700 rounded">
                <p class="font-semibold">{{ training.name }}</p>
                <div class="w-full bg-gray-600 rounded-full h-2 mt-2">
                    <div class="bg-purple-500 h-2 rounded-full"
                         style="width: {{ training.progress }}%"></div>
                </div>
                <p class="text-sm mt-1">Epoch {{ training.current_epoch }}/{{ training.total_epochs }}</p>
            </div>
            {% empty %}
            <p class="text-gray-400">No active trainings</p>
            {% endfor %}
        </div>
    </div>

    <!-- Recent Models Card -->
    <div class="bg-gray-800 rounded-lg p-6">
        <h2 class="text-xl font-bold mb-4 flex items-center">
            <i class="fas fa-database mr-2 text-blue-500"></i>
            Recent Models
        </h2>
        <div class="space-y-2">
            {% for model in recent_models %}
            <div class="flex justify-between items-center p-2 hover:bg-gray-700 rounded">
                <span>{{ model.name }}</span>
                <span class="text-sm text-gray-400">Loss: {{ model.best_loss|floatformat:3 }}</span>
            </div>
            {% empty %}
            <p class="text-gray-400">No models yet</p>
            {% endfor %}
        </div>
    </div>
</div>

<!-- Training Metrics Chart -->
<div class="mt-8 bg-gray-800 rounded-lg p-6">
    <h2 class="text-xl font-bold mb-4">Training Metrics</h2>
    <canvas id="metricsChart" width="400" height="100"></canvas>
</div>

<script>
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/dashboard/');

ws.onmessage = function(e) {
    const data = JSON.parse(e.data);
    updateDashboard(data);
};

function updateDashboard(data) {
    // Update GPU stats
    if (data.gpu) {
        document.getElementById('gpu-memory').textContent = data.gpu.memory;
        document.getElementById('gpu-temp').textContent = data.gpu.temp;
        document.getElementById('gpu-util').textContent = data.gpu.util;
    }

    // Update training progress
    if (data.training) {
        // Update progress bars and metrics
    }
}

// Initialize Chart.js
const ctx = document.getElementById('metricsChart').getContext('2d');
const metricsChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Train Loss',
            data: [],
            borderColor: 'rgb(139, 92, 246)',
            tension: 0.1
        }, {
            label: 'Val Loss',
            data: [],
            borderColor: 'rgb(59, 130, 246)',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: false
            }
        }
    }
});
</script>
{% endblock %}'''

    with open("templates/dashboard/index.html", "w") as f:
        f.write(dashboard_template)

    print("[OK] Templates created")

    # Create initial views
    create_initial_views()

    # Create URLs
    create_urls()

    print("\n" + "=" * 60)
    print("[SUCCESS] Django project setup complete!")
    print("\nNext steps:")
    print("1. Update settings.py with your configuration")
    print("2. Run: python manage.py migrate")
    print("3. Create superuser: python manage.py createsuperuser")
    print("4. Run server: python manage.py runserver")
    print("5. Visit: http://localhost:8000")

def create_initial_views():
    """Create initial view files"""

    # Dashboard views
    dashboard_views = '''from django.shortcuts import render
from django.views.generic import TemplateView
import torch
import nvidia_ml_py as nvml

class DashboardView(TemplateView):
    template_name = 'dashboard/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get GPU info
        if torch.cuda.is_available():
            context['gpu_name'] = torch.cuda.get_device_name(0)
            context['gpu_memory'] = f"{torch.cuda.memory_allocated(0) / 1e9:.1f}GB"
        else:
            context['gpu_name'] = "No GPU detected"
            context['gpu_memory'] = "N/A"

        # TODO: Get active trainings from database
        context['active_trainings'] = []

        # TODO: Get recent models from database
        context['recent_models'] = []

        return context
'''

    with open("dashboard/views.py", "w") as f:
        f.write(dashboard_views)

    print("[OK] Views created")

def create_urls():
    """Create URL configurations"""

    # Main URLs
    main_urls = '''from django.contrib import admin
from django.urls import path, include
from dashboard.views import DashboardView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', DashboardView.as_view(), name='dashboard'),
    path('api/', include('api.urls')),
    path('training/', include('training.urls')),
    path('generation/', include('generation.urls')),
]
'''

    with open("robo_poet_web/urls.py", "w") as f:
        f.write(main_urls)

    # API URLs
    api_urls = '''from django.urls import path
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
# TODO: Register viewsets

urlpatterns = router.urls
'''

    with open("api/urls.py", "w") as f:
        f.write(api_urls)

    # Training URLs
    training_urls = '''from django.urls import path

urlpatterns = [
    # TODO: Add training URLs
]
'''

    with open("training/urls.py", "w") as f:
        f.write(training_urls)

    # Generation URLs
    with open("generation/urls.py", "w") as f:
        f.write(training_urls)

    print("[OK] URLs configured")

if __name__ == "__main__":
    setup_django_project()