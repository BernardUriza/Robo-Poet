from django.urls import path
from .views import TrainingControlView, GPUStatusView

urlpatterns = [
    path('api/sessions/', TrainingControlView.as_view(), name='training-sessions'),
    path('api/gpu-status/', GPUStatusView.as_view(), name='gpu-status'),
]