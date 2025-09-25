import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import TrainingSession, TrainingMetric
import subprocess
import os


class TrainingConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time training metrics"""

    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs'].get('session_id', 'global')
        self.room_group_name = f'training_{self.session_id}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        # Send initial data
        await self.send_initial_data()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        data = json.loads(text_data)
        message_type = data.get('type')

        if message_type == 'start_training':
            await self.start_training(data)
        elif message_type == 'stop_training':
            await self.stop_training(data)
        elif message_type == 'pause_training':
            await self.pause_training(data)
        elif message_type == 'get_metrics':
            await self.send_metrics()
        elif message_type == 'get_gpu_status':
            await self.send_gpu_status()

    async def start_training(self, data):
        """Start a new training session via robo_poet.py"""
        model_name = data.get('model_name', 'web_model')
        cycles = data.get('cycles', 1)
        epochs = data.get('epochs', 10)
        phase = data.get('phase', '3')  # Default to Phase 3 (Intelligent Cycle)

        # Create training session in database
        session = await self.create_training_session(model_name, cycles, epochs)

        # Build robo_poet.py command
        cmd = [
            'python', 'robo_poet.py',
            '--headless'  # Run without interactive menu
        ]

        # Set environment for Claude API key and Django integration
        env = os.environ.copy()
        env['DJANGO_RUN'] = 'true'
        env['TRAINING_SESSION_ID'] = str(session.id)
        if 'CLAUDE_API_KEY' in env:
            # API key already in environment
            pass

        # Create input for robo_poet.py
        input_data = f"{phase}\n{model_name}\n{cycles}\n{epochs}\n\n"

        # Start robo_poet.py process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1
        )

        # Store process ID
        await self.update_session_process(session.id, process.pid)

        # Send input
        process.stdin.write(input_data)
        process.stdin.flush()

        # Monitor process output
        asyncio.create_task(self.monitor_training_process(process, session.id))

        # Send confirmation
        await self.send(text_data=json.dumps({
            'type': 'training_started',
            'session_id': session.id,
            'process_id': process.pid
        }))

    async def monitor_training_process(self, process, session_id):
        """Monitor robo_poet.py output and send updates"""
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                # Parse output for metrics
                metrics = self.parse_training_output(line)
                if metrics:
                    # Save to database
                    await self.save_metrics(session_id, metrics)

                    # Send to WebSocket
                    await self.channel_layer.group_send(
                        self.room_group_name,
                        {
                            'type': 'training_update',
                            'metrics': metrics
                        }
                    )

            await asyncio.sleep(0.1)

    def parse_training_output(self, line):
        """Parse robo_poet.py output for metrics"""
        metrics = {}

        # Look for patterns in output
        if 'Epoch' in line and 'Loss' in line:
            # Extract epoch and loss
            import re
            epoch_match = re.search(r'Epoch\s+(\d+)', line)
            loss_match = re.search(r'Loss:\s+([\d.]+)', line)

            if epoch_match and loss_match:
                metrics['epoch'] = int(epoch_match.group(1))
                metrics['train_loss'] = float(loss_match.group(1))

        elif 'Perplexity' in line:
            import re
            perp_match = re.search(r'Perplexity:\s+([\d.]+)', line)
            if perp_match:
                metrics['perplexity'] = float(perp_match.group(1))

        elif 'GPU Memory' in line:
            import re
            mem_match = re.search(r'GPU Memory:\s+([\d.]+)\s*GB', line)
            if mem_match:
                metrics['gpu_memory'] = float(mem_match.group(1))

        elif 'Claude AI' in line and 'suggestion' in line.lower():
            metrics['claude_suggestion'] = line

        return metrics if metrics else None

    async def stop_training(self, data):
        """Stop a training session"""
        session_id = data.get('session_id')
        if session_id:
            session = await self.get_session(session_id)
            if session and session.process_id:
                # Kill the process
                try:
                    os.kill(session.process_id, 9)  # SIGKILL
                    await self.update_session_status(session_id, 'failed')
                    await self.send(text_data=json.dumps({
                        'type': 'training_stopped',
                        'session_id': session_id
                    }))
                except ProcessLookupError:
                    pass

    async def training_update(self, event):
        """Send training update to WebSocket"""
        metrics = event['metrics']
        await self.send(text_data=json.dumps({
            'type': 'metrics_update',
            'metrics': metrics
        }))

    async def send_gpu_status(self):
        """Send current GPU status"""
        import torch
        if torch.cuda.is_available():
            gpu_status = {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0) / 1e9,
                'memory_reserved': torch.cuda.memory_reserved(0) / 1e9,
                'utilization': 0  # Would need nvidia-ml-py for real utilization
            }
        else:
            gpu_status = {
                'name': 'No GPU detected',
                'memory_allocated': 0,
                'memory_reserved': 0,
                'utilization': 0
            }

        await self.send(text_data=json.dumps({
            'type': 'gpu_status',
            'data': gpu_status
        }))

    async def send_initial_data(self):
        """Send initial dashboard data"""
        sessions = await self.get_active_sessions()
        await self.send(text_data=json.dumps({
            'type': 'initial_data',
            'active_sessions': sessions
        }))

    async def send_metrics(self):
        """Send recent metrics for a session"""
        if self.session_id != 'global':
            metrics = await self.get_session_metrics(self.session_id)
            await self.send(text_data=json.dumps({
                'type': 'metrics_data',
                'metrics': metrics
            }))

    # Database operations
    @database_sync_to_async
    def create_training_session(self, name, cycles, epochs):
        return TrainingSession.objects.create(
            name=name,
            total_cycles=cycles,
            total_epochs=epochs,
            status='running',
            claude_enabled=True
        )

    @database_sync_to_async
    def update_session_process(self, session_id, process_id):
        session = TrainingSession.objects.get(id=session_id)
        session.process_id = process_id
        session.save()

    @database_sync_to_async
    def update_session_status(self, session_id, status):
        session = TrainingSession.objects.get(id=session_id)
        session.status = status
        session.save()

    @database_sync_to_async
    def save_metrics(self, session_id, metrics_data):
        session = TrainingSession.objects.get(id=session_id)

        # Update session with latest metrics
        if 'train_loss' in metrics_data:
            session.current_loss = metrics_data['train_loss']
            if not session.best_loss or metrics_data['train_loss'] < session.best_loss:
                session.best_loss = metrics_data['train_loss']

        if 'epoch' in metrics_data:
            session.current_epoch = metrics_data['epoch']

        if 'perplexity' in metrics_data:
            session.perplexity = metrics_data['perplexity']

        session.save()

        # Create metric record
        TrainingMetric.objects.create(
            session=session,
            epoch=metrics_data.get('epoch', session.current_epoch),
            train_loss=metrics_data.get('train_loss', 0),
            perplexity=metrics_data.get('perplexity'),
            gpu_memory_used=metrics_data.get('gpu_memory')
        )

    @database_sync_to_async
    def get_session(self, session_id):
        try:
            return TrainingSession.objects.get(id=session_id)
        except TrainingSession.DoesNotExist:
            return None

    @database_sync_to_async
    def get_active_sessions(self):
        sessions = TrainingSession.objects.filter(status='running').values(
            'id', 'name', 'current_epoch', 'total_epochs',
            'current_loss', 'perplexity'
        )
        return list(sessions)

    @database_sync_to_async
    def get_session_metrics(self, session_id):
        metrics = TrainingMetric.objects.filter(session_id=session_id).values(
            'epoch', 'train_loss', 'val_loss', 'perplexity',
            'gpu_memory_used', 'recorded_at'
        ).order_by('-epoch')[:50]  # Last 50 metrics
        return list(metrics)