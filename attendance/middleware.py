import django.utils.timezone as tz
from .models import SystemSetting

class TimezoneMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Dynamic timezone setting was removed from SystemSetting model.
        # System will now use the default TIME_ZONE from settings.py.
        tz.deactivate()
        
        response = self.get_response(request)
        return response
