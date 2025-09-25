from django.shortcuts import render
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
