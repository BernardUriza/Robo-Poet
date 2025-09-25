from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .models import TrainingSession, TrainingMetric
import subprocess
import json
import os


@method_decorator(csrf_exempt, name='dispatch')
class TrainingControlView(View):
    """API endpoint to control training sessions"""

    def get(self, request):
        """Get list of training sessions"""
        sessions = TrainingSession.objects.all().values(
            'id', 'name', 'status', 'current_epoch', 'total_epochs',
            'current_loss', 'perplexity', 'created_at'
        ).order_by('-created_at')[:20]

        return JsonResponse({'sessions': list(sessions)})

    def post(self, request):
        """Start a new training session"""
        data = json.loads(request.body)

        model_name = data.get('model_name', 'web_model')
        cycles = data.get('cycles', 1)
        epochs = data.get('epochs', 10)
        phase = data.get('phase', '3')  # Default to Phase 3

        # Create session
        session = TrainingSession.objects.create(
            name=model_name,
            total_cycles=cycles,
            total_epochs=epochs,
            phase=f'Phase{phase}',
            status='pending',
            claude_enabled=(phase == '3')
        )

        # Prepare robo_poet.py command
        env = os.environ.copy()
        cmd = ['python', 'robo_poet.py']

        # Start process
        input_data = f"{phase}\n{model_name}\n{cycles}\n{epochs}\n\n"

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            # Send input
            process.stdin.write(input_data)
            process.stdin.flush()
            process.stdin.close()

            # Update session with process ID
            session.process_id = process.pid
            session.status = 'running'
            session.save()

            return JsonResponse({
                'success': True,
                'session_id': session.id,
                'process_id': process.pid
            })

        except Exception as e:
            session.status = 'failed'
            session.save()
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)


class GPUStatusView(View):
    """Get current GPU status"""

    def get(self, request):
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0) / 1e9,
                'memory_reserved': torch.cuda.memory_reserved(0) / 1e9,
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        else:
            gpu_info = {
                'available': False,
                'name': 'No GPU detected'
            }

        return JsonResponse(gpu_info)
