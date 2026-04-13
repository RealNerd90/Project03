from .models import SystemSetting

def system_settings(request):
    """Provides system settings to all templates."""
    setting = SystemSetting.objects.first()
    if not setting:
        setting = SystemSetting.objects.create() # Create default if missing
    return {
        'system_settings': setting
    }
