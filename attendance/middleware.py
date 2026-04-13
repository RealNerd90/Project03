import django.utils.timezone as tz
from .models import SystemSetting

class TimezoneMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        setting = SystemSetting.objects.first()
        if setting and setting.timezone:
            try:
                tz.activate(setting.timezone)
            except Exception:
                tz.deactivate()
        else:
            tz.deactivate()
        
        response = self.get_response(request)
        return response
