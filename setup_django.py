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
        "gunicorn==21.2.0",
        "channels==4.0.0",
        "channels-redis==4.2.0",
        "daphne==4.1.0"
    ]

    for package in requirements:
        print(f"Installing {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Could not install {package}")

    # Create Django project using module approach
    print("\n[2/5] Creating Django project structure...")

    # Check if manage.py already exists
    manage_py = Path("manage.py")
    if not manage_py.exists():
        # Try to create project using Django as a module
        try:
            import django
            from django.core.management import execute_from_command_line

            # Create project
            execute_from_command_line(['django-admin', 'startproject', 'robo_poet_web', '.'])
            print("[OK] Django project created")
        except Exception as e:
            print(f"[ERROR] Could not create Django project: {e}")
            print("\nTrying alternative method...")

            # Alternative: create project structure manually
            create_django_structure_manually()
    else:
        print("[INFO] Django project already exists")

    # Create apps
    print("\n[3/5] Creating Django apps...")
    apps = ["training", "generation", "api", "dashboard"]

    for app in apps:
        app_path = Path(app)
        if not app_path.exists():
            try:
                subprocess.run([sys.executable, "manage.py", "startapp", app])
                print(f"[OK] Created app: {app}")
            except:
                # Create app structure manually
                create_app_manually(app)
                print(f"[OK] Manually created app: {app}")
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

    # Update settings
    update_settings()

    print("\n" + "=" * 60)
    print("[SUCCESS] Django project setup complete!")
    print("\nNext steps:")
    print("1. Run: python manage.py migrate")
    print("2. Create superuser: python manage.py createsuperuser")
    print("3. Run server: python manage.py runserver")
    print("4. Visit: http://localhost:8000")

def create_django_structure_manually():
    """Create Django project structure manually if django-admin fails"""

    # Create manage.py
    manage_content = '''#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'robo_poet_web.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
'''

    with open("manage.py", "w") as f:
        f.write(manage_content)

    # Create robo_poet_web directory
    web_dir = Path("robo_poet_web")
    web_dir.mkdir(exist_ok=True)

    # Create __init__.py
    (web_dir / "__init__.py").touch()

    # Create settings.py
    settings_content = '''from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-development-key-change-in-production'

DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'channels',
    'dashboard',
    'training',
    'generation',
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'robo_poet_web.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'robo_poet_web.wsgi.application'
ASGI_APPLICATION = 'robo_poet_web.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer'
    }
}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
'''

    with open(web_dir / "settings.py", "w") as f:
        f.write(settings_content)

    # Create wsgi.py
    wsgi_content = '''import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'robo_poet_web.settings')
application = get_wsgi_application()
'''

    with open(web_dir / "wsgi.py", "w") as f:
        f.write(wsgi_content)

    # Create asgi.py
    asgi_content = '''import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'robo_poet_web.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    # WebSocket handler will be added later
})
'''

    with open(web_dir / "asgi.py", "w") as f:
        f.write(asgi_content)

    print("[OK] Django project structure created manually")

def create_app_manually(app_name):
    """Create Django app structure manually"""
    app_dir = Path(app_name)
    app_dir.mkdir(exist_ok=True)

    # Create basic app files
    (app_dir / "__init__.py").touch()
    (app_dir / "apps.py").write_text(f'''from django.apps import AppConfig

class {app_name.capitalize()}Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = '{app_name}'
''')

    (app_dir / "models.py").write_text('''from django.db import models

# Create your models here.
''')

    (app_dir / "views.py").write_text('''from django.shortcuts import render

# Create your views here.
''')

    (app_dir / "admin.py").write_text('''from django.contrib import admin

# Register your models here.
''')

    (app_dir / "tests.py").write_text('''from django.test import TestCase

# Create your tests here.
''')

    # Create migrations directory
    migrations_dir = app_dir / "migrations"
    migrations_dir.mkdir(exist_ok=True)
    (migrations_dir / "__init__.py").touch()

def create_initial_views():
    """Create initial view files"""

    # Dashboard views
    dashboard_views = '''from django.shortcuts import render
from django.views.generic import TemplateView
import torch

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
    path('training/', DashboardView.as_view(), name='training'),
    path('generation/', DashboardView.as_view(), name='generation'),
    path('models/', DashboardView.as_view(), name='models'),
]
'''

    with open("robo_poet_web/urls.py", "w") as f:
        f.write(main_urls)

    print("[OK] URLs configured")

def update_settings():
    """Update settings file if needed"""
    settings_path = Path("robo_poet_web/settings.py")
    if settings_path.exists():
        print("[OK] Settings file exists")
    else:
        print("[WARNING] Settings file not found, creating manually")
        create_django_structure_manually()

if __name__ == "__main__":
    setup_django_project()