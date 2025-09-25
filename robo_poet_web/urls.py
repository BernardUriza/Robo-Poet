from django.contrib import admin
from django.urls import path, include
from dashboard.views import DashboardView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', DashboardView.as_view(), name='dashboard'),
    path('training/', include('training.urls')),
    path('generation/', DashboardView.as_view(), name='generation'),
    path('models/', DashboardView.as_view(), name='models'),
]
