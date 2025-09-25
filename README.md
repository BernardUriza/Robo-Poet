# 🤖 Robo-Poet: AI Text Generation Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Django](https://img.shields.io/badge/Django-5.0+-green.svg)](https://www.djangoproject.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-Academic-blue.svg)](LICENSE)

An advanced academic text generation framework featuring real-time training monitoring, Claude AI integration, and enterprise-grade architecture. Built for GPU-accelerated transformer training on literary corpora.

## 🚀 Key Features

### 🌐 **Django Web Interface**
- **Real-time Training Dashboard** - Live metrics, GPU monitoring, progress visualization
- **WebSocket Integration** - Instant updates via Django Channels
- **Interactive Training Control** - Start/stop training sessions from the web
- **Chart.js Visualizations** - Loss curves, perplexity tracking, performance graphs
- **Session Management** - Complete training history and model tracking

### 🧠 **Intelligent Training System**
- **Claude AI Integration** - Phase 3 intelligent training with AI-guided optimization
- **Adaptive Learning** - Real-time dataset improvements based on performance
- **Multi-phase Training** - Intensive training, generation, and intelligent cycles
- **GPU Optimization** - RTX 2000 Ada specific optimizations with mixed precision

### 🏗️ **Enterprise Architecture**
- **Clean Architecture** - Domain-driven design with separation of concerns
- **Modular Components** - Orchestrator pattern with pluggable interfaces
- **Dual-mode Operation** - Web interface + CLI for different use cases
- **Comprehensive Testing** - Academic validation and integration test suites

## 📋 Quick Start

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA 11.8+ (recommended)
- 8GB+ VRAM for optimal performance
- WSL2 + Windows 11 (recommended) or Linux

### 1. Installation
```bash
git clone https://github.com/BernardUriza/Robo-Poet.git
cd Robo-Poet

# Install dependencies
pip install -r requirements.txt

# Install additional packages for web interface
pip install channels channels-redis daphne websocket-client
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Configure your Claude AI API key (optional)
# Edit .env and set: CLAUDE_API_KEY=your-api-key-here
```

### 3. Database Setup
```bash
# Run Django migrations
python manage.py makemigrations
python manage.py migrate
```

### 4. Launch Web Interface
```bash
# Start Django development server
python manage.py runserver 8000

# Open browser and navigate to:
# http://localhost:8000
```

### 5. Start Training
- **Web Interface**: Use the dashboard to configure and start training
- **CLI Mode**: Run `python robo_poet.py` for interactive training

## 🎯 Usage Examples

### Web-based Training
```bash
# Start the web server
python manage.py runserver 8000

# Navigate to http://localhost:8000
# Use the dashboard to:
# 1. Configure model parameters
# 2. Start training sessions
# 3. Monitor real-time metrics
# 4. View training history
```

### CLI Training (Advanced)
```bash
# Interactive menu system
python robo_poet.py

# Direct training mode
python robo_poet.py --model my_model --epochs 25

# Intelligent cycle with Claude AI
python robo_poet.py
# Select option 3: 🧠 FASE 3: Ciclo Inteligente con Claude AI

# Testing and validation
python robo_poet.py --test quick
```

### Text Generation
```bash
# Web interface: Use generation panel in dashboard
# CLI mode: python robo_poet.py → Select Phase 2: Generation
# Direct: python robo_poet.py --generate model.keras --seed "Once upon a time"
```

## 🏛️ Architecture Overview

```
Robo-Poet/
├── 🌐 Django Web Framework          # Primary interface
│   ├── dashboard/                   # Main dashboard app
│   ├── training/                    # Training management
│   ├── api/                         # REST API endpoints
│   ├── templates/                   # Web templates
│   └── robo_poet_web/              # Django project settings
├── 🧠 Core AI System               # Training engine
│   ├── src/orchestrator.py         # Main orchestrator
│   ├── src/models/                  # Neural network models
│   ├── src/intelligence/            # Claude AI integration
│   └── src/interface/               # Phase interfaces
├── 📊 Data & Configuration
│   ├── corpus/                      # Training text data
│   ├── config/                      # Configuration files
│   └── training/                    # Generated models
└── 🧪 Testing & Scripts
    ├── test_training_integration.py # Integration tests
    └── scripts/                     # Utility scripts
```

## 🔧 Technical Specifications

### Model Architecture
- **Type**: MinGPT-inspired transformer
- **Parameters**: <10M (academic demonstrations)
- **Layers**: 6 transformer layers
- **Heads**: 8 attention heads
- **Embedding**: 256 dimensions
- **Context Window**: 128 tokens (configurable)

### Performance Targets
- **Validation Loss**: <5.0 for coherent generation
- **Training Time**: 30-60 minutes on RTX 2000 Ada
- **Generation Quality**: 200+ coherent tokens
- **GPU Memory**: <8GB VRAM usage

### Technology Stack
- **Backend**: Django 5.0, Django Channels, PyTorch 2.0+
- **Frontend**: HTML5, TailwindCSS, Chart.js, WebSockets
- **AI Integration**: Claude API (Haiku model)
- **Database**: SQLite (development), PostgreSQL (production)
- **Deployment**: Daphne ASGI server, Redis (production)

## 📊 Real-time Features

### WebSocket Endpoints
```javascript
// Training metrics WebSocket
ws://localhost:8000/ws/training/global/

// Example WebSocket message
{
  "type": "metrics_update",
  "metrics": {
    "epoch": 15,
    "train_loss": 3.245,
    "perplexity": 25.8,
    "gpu_memory": 4.2
  }
}
```

### REST API
```bash
# Get training sessions
GET /training/api/sessions/

# Start new training
POST /training/api/sessions/
{
  "model_name": "shakespeare_model",
  "epochs": 20,
  "cycles": 3,
  "phase": "3"
}

# GPU status
GET /training/api/gpu-status/
```

## 🧪 Testing

### Integration Testing
```bash
# Run complete integration test suite
python test_training_integration.py

# Expected output:
# gpu_status: [PASS]
# http_api: [PASS]
# websocket: [PASS]
# django_integration: [PASS]
```

### Academic Validation
```bash
# Quick validation (5 minutes)
python robo_poet.py --test quick

# Full academic test suite (20 minutes)
python robo_poet.py --test full

# Selective testing
python robo_poet.py --test selective --test-selection training gradient_analysis
```

## 🤖 Claude AI Integration

### Setup
1. Get your Claude API key from [Anthropic Console](https://console.anthropic.com/)
2. Set in `.env`: `CLAUDE_API_KEY=your-api-key-here`
3. Configure model: `CLAUDE_MODEL=claude-3-haiku-20240307`

### Features
- **Intelligent Training Cycles** - AI analyzes training progress and suggests improvements
- **Dataset Optimization** - Real-time recommendations for data preprocessing
- **Hyperparameter Tuning** - AI-guided parameter adjustments
- **Cost Efficient** - ~$0.25 per full training cycle with Haiku model

### Usage
```bash
# Start intelligent training
python robo_poet.py
# Select: 3 → 🧠 FASE 3: Ciclo Inteligente con Claude AI

# Web interface: Enable "Intelligent Cycle" in training settings
```

## 📈 Performance Monitoring

### GPU Metrics
- **Real-time Memory Usage** - Live VRAM consumption tracking
- **Temperature Monitoring** - GPU thermal status
- **Utilization Graphs** - Compute usage over time

### Training Metrics
- **Loss Visualization** - Real-time training/validation loss curves
- **Perplexity Tracking** - Model quality indicators
- **Gradient Analysis** - Training stability monitoring
- **Learning Rate Scheduling** - Adaptive optimization

## 🛠️ Development

### Adding New Features
1. **Web Interface**: Add Django apps in project root
2. **Training Logic**: Extend `src/interface/` phases
3. **Models**: Add to `src/models/` directory
4. **WebSocket Events**: Extend `training/consumers.py`

### Custom Training Phases
```python
# Example: Add Phase 4 in src/interface/phase4_custom.py
class Phase4CustomTraining:
    def run(self):
        # Your custom training logic
        pass

    def run_django_mode(self, model_name, cycles, epochs):
        # Django integration
        pass
```

### Database Models
```python
# Extend training/models.py for custom metrics
class CustomMetric(models.Model):
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    metric_name = models.CharField(max_length=100)
    value = models.FloatField()
    recorded_at = models.DateTimeField(auto_now_add=True)
```

## 🚀 Deployment

### Development
```bash
python manage.py runserver 8000
```

### Production
```bash
# Install production dependencies
pip install gunicorn redis

# Run with Daphne (ASGI)
daphne -p 8000 robo_poet_web.asgi:application

# Redis for channel layers (production)
# Update CHANNEL_LAYERS in settings.py
```

## 📚 Documentation

- **[Django Architecture](docs/DJANGO_ARCHITECTURE.md)** - Web framework details
- **[Claude AI Integration](docs/CLAUDE_AI_INTEGRATION.md)** - AI features guide
- **[CLAUDE.md](CLAUDE.md)** - Claude Code assistant guidance

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines
- Follow clean architecture patterns
- Write comprehensive tests
- Document API changes
- Maintain CRLF line endings (Windows)
- Use conventional commit messages

## 📄 License

This project is licensed under Academic License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **Django Software Foundation** - Web framework
- **Anthropic** - Claude AI integration
- **NVIDIA** - CUDA acceleration support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/BernardUriza/Robo-Poet/issues)
- **Documentation**: Check `docs/` directory
- **Testing**: Run `python test_training_integration.py`

---

**Built with ❤️ for Academic AI Research**

*Optimized for NVIDIA RTX 2000 Ada | WSL2 + Windows 11 | Academic Use*