from django.db import models
from django.utils import timezone
import json

class TrainingSession(models.Model):
    """Model for tracking training sessions from robo_poet.py"""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('paused', 'Paused')
    ]

    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, default='GPT')
    phase = models.CharField(max_length=50, default='Phase3')  # Intelligent Cycle

    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Configuration
    config = models.JSONField(default=dict)
    dataset_path = models.CharField(max_length=500, blank=True)

    # Progress tracking
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=10)
    current_cycle = models.IntegerField(default=0)
    total_cycles = models.IntegerField(default=1)

    # Performance metrics
    current_loss = models.FloatField(null=True, blank=True)
    best_loss = models.FloatField(null=True, blank=True)
    perplexity = models.FloatField(null=True, blank=True)

    # GPU Information
    gpu_name = models.CharField(max_length=200, blank=True)
    gpu_memory_used = models.FloatField(null=True, blank=True)  # in GB

    # Claude AI Integration
    claude_enabled = models.BooleanField(default=False)
    claude_suggestions = models.JSONField(default=list)
    claude_cost = models.FloatField(default=0.0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Process ID for controlling robo_poet.py
    process_id = models.IntegerField(null=True, blank=True)

    def duration(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def progress_percentage(self):
        if self.total_epochs > 0:
            return (self.current_epoch / self.total_epochs) * 100
        return 0

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.status})"


class TrainingMetric(models.Model):
    """Individual metrics recorded during training"""

    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='metrics')

    epoch = models.IntegerField()
    cycle = models.IntegerField(default=1)
    batch = models.IntegerField(null=True, blank=True)

    # Losses
    train_loss = models.FloatField()
    val_loss = models.FloatField(null=True, blank=True)

    # Performance
    perplexity = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(null=True, blank=True)

    # GPU Stats
    gpu_memory_used = models.FloatField(null=True, blank=True)
    gpu_utilization = models.FloatField(null=True, blank=True)
    gpu_temperature = models.FloatField(null=True, blank=True)

    # Timing
    epoch_time = models.FloatField(null=True, blank=True)  # seconds

    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['session', 'cycle', 'epoch', 'batch']

    def __str__(self):
        return f"Session {self.session.id} - Epoch {self.epoch}"
