# 🚀 Django Architecture for Robo-Poet Web Interface

## Executive Summary
Transform Robo-Poet from CLI to professional web application with real-time training monitoring, interactive text generation, and Claude AI integration.

## 🏗️ Proposed Architecture

### 1. Core Stack
```
Frontend:
├── React/Vue.js (real-time dashboard)
├── Chart.js/D3.js (training visualizations)
└── WebSocket client (live updates)

Backend:
├── Django 5.0+ (main framework)
├── Django REST Framework (API)
├── Django Channels (WebSockets)
├── Celery (async tasks)
└── Redis (cache & message broker)

ML Pipeline:
├── PyTorch (training)
├── TensorBoard (metrics)
└── Claude AI API (intelligence)
```

## 📊 Key Features

### 1. Real-Time Training Dashboard
```python
# views.py
class TrainingDashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'training/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_trainings'] = Training.objects.filter(status='running')
        context['gpu_status'] = GPUMonitor.get_status()
        return context
```

### 2. WebSocket Live Metrics
```python
# consumers.py
class TrainingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.training_id = self.scope['url_route']['kwargs']['training_id']
        await self.channel_layer.group_add(
            f'training_{self.training_id}',
            self.channel_name
        )
        await self.accept()

    async def receive(self, text_data):
        # Handle incoming messages
        data = json.loads(text_data)
        if data['type'] == 'get_metrics':
            metrics = await self.get_training_metrics(self.training_id)
            await self.send(text_data=json.dumps(metrics))

    async def training_update(self, event):
        # Send training updates to WebSocket
        await self.send(text_data=json.dumps({
            'epoch': event['epoch'],
            'loss': event['loss'],
            'perplexity': event['perplexity'],
            'gpu_usage': event['gpu_usage']
        }))
```

### 3. Celery Background Training
```python
# tasks.py
@shared_task
def train_model_task(model_id, config):
    """Async training task"""
    model = Model.objects.get(id=model_id)

    # Initialize PyTorch model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_model = GPT(config)

    # Training loop with real-time updates
    for epoch in range(config['epochs']):
        loss = train_epoch(gpt_model, train_loader)

        # Send real-time update via WebSocket
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f'training_{model_id}',
            {
                'type': 'training_update',
                'epoch': epoch,
                'loss': loss.item(),
                'perplexity': math.exp(loss.item())
            }
        )

        # Save checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)

    return model_id
```

## 🎨 User Interface Components

### 1. Training Control Panel
- Start/Stop/Pause training
- Real-time loss graphs
- GPU memory usage
- Checkpoint management
- Dataset selection

### 2. Text Generation Studio
- Interactive prompt interface
- Temperature/Top-k controls
- Real-time generation
- Multiple model comparison
- Style presets (Shakespeare, Alice, etc.)

### 3. Claude AI Integration Panel
- Dataset improvement suggestions
- Automatic cycle configuration
- Cost tracking
- API usage monitoring

### 4. Model Management
- List all models
- Download checkpoints
- Version control
- A/B testing interface

## 🔥 Advanced Features

### 1. TensorBoard Integration
```python
# utils/tensorboard.py
class TensorBoardManager:
    def __init__(self):
        self.writer = SummaryWriter('runs/robo_poet')

    def log_metrics(self, epoch, train_loss, val_loss, perplexity):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Metrics/perplexity', perplexity, epoch)

    def embed_tensorboard(self):
        """Embed TensorBoard in Django template"""
        return f'<iframe src="http://localhost:6006" width="100%" height="800"></iframe>'
```

### 2. GPU Monitoring
```python
# utils/gpu_monitor.py
import nvidia_ml_py as nvml

class GPUMonitor:
    @staticmethod
    def get_real_time_stats():
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)

        return {
            'temperature': nvml.nvmlDeviceGetTemperature(handle, 0),
            'memory_used': nvml.nvmlDeviceGetMemoryInfo(handle).used / 1e9,
            'memory_total': nvml.nvmlDeviceGetMemoryInfo(handle).total / 1e9,
            'utilization': nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            'power_draw': nvml.nvmlDeviceGetPowerUsage(handle) / 1000
        }
```

### 3. Model Comparison
```python
# views.py
class ModelComparisonView(View):
    def post(self, request):
        models = request.POST.getlist('models')
        prompt = request.POST.get('prompt')

        results = []
        for model_id in models:
            model = load_model(model_id)
            generated_text = generate(model, prompt)
            perplexity = calculate_perplexity(model, prompt)

            results.append({
                'model': model_id,
                'text': generated_text,
                'perplexity': perplexity,
                'inference_time': measure_time()
            })

        return JsonResponse({'comparisons': results})
```

## 📦 Database Models

```python
# models.py
class TrainingSession(models.Model):
    name = models.CharField(max_length=200)
    dataset = models.ForeignKey('Dataset', on_delete=models.CASCADE)
    model_config = models.JSONField()
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    gpu_used = models.CharField(max_length=100)
    total_epochs = models.IntegerField()
    current_epoch = models.IntegerField(default=0)
    best_loss = models.FloatField(null=True)

class Checkpoint(models.Model):
    training = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    epoch = models.IntegerField()
    train_loss = models.FloatField()
    val_loss = models.FloatField()
    perplexity = models.FloatField()
    file_path = models.FileField(upload_to='checkpoints/')
    created_at = models.DateTimeField(auto_now_add=True)
    is_best = models.BooleanField(default=False)

class GeneratedText(models.Model):
    checkpoint = models.ForeignKey(Checkpoint, on_delete=models.CASCADE)
    prompt = models.TextField()
    generated = models.TextField()
    temperature = models.FloatField()
    top_k = models.IntegerField()
    top_p = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
```

## 🚀 Implementation Plan

### Phase 1: Core Setup (Week 1)
- [ ] Django project initialization
- [ ] Database models
- [ ] Basic REST API
- [ ] User authentication

### Phase 2: Training Integration (Week 2)
- [ ] Celery setup
- [ ] PyTorch training tasks
- [ ] Checkpoint management
- [ ] Progress tracking

### Phase 3: Real-Time Features (Week 3)
- [ ] Django Channels setup
- [ ] WebSocket consumers
- [ ] Live metric streaming
- [ ] GPU monitoring

### Phase 4: UI Development (Week 4)
- [ ] React/Vue dashboard
- [ ] Real-time charts
- [ ] Text generation interface
- [ ] Model comparison tools

### Phase 5: Advanced Features (Week 5)
- [ ] TensorBoard integration
- [ ] Claude AI panel
- [ ] A/B testing
- [ ] Export/Import models

## 💰 Benefits

1. **Professional Interface**: Modern web UI instead of CLI
2. **Remote Access**: Train and monitor from anywhere
3. **Multi-User**: Teams can collaborate on training
4. **Real-Time Monitoring**: Live metrics and GPU stats
5. **Better Organization**: Database-backed model management
6. **Scalability**: Ready for production deployment
7. **API First**: Can integrate with other tools
8. **Mobile Friendly**: Responsive design for all devices

## 🛠️ Required Dependencies

```txt
# requirements_django.txt
Django==5.0.6
djangorestframework==3.15.1
django-channels==4.3.0
channels-redis==4.2.0
celery==5.3.4
redis==5.0.4
django-cors-headers==4.3.1
daphne==4.1.0
nvidia-ml-py==12.535.108
django-extensions==3.2.3
django-debug-toolbar==4.3.0
```

## 🔌 Frontend Stack

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "axios": "^1.6.8",
    "chart.js": "^4.4.2",
    "react-chartjs-2": "^5.2.0",
    "socket.io-client": "^4.7.0",
    "tailwindcss": "^3.4.1"
  }
}
```

## 🎯 Next Steps

1. **Approve architecture**: Review and approve this proposal
2. **Setup Django project**: Initialize with proper structure
3. **Create REST API**: For model training control
4. **Implement WebSockets**: For real-time updates
5. **Build dashboard**: Modern UI with live charts
6. **Deploy**: Docker + Nginx + Gunicorn/Daphne

This architecture transforms Robo-Poet into a production-ready ML platform with professional web interface, real-time monitoring, and team collaboration features.