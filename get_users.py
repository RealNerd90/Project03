import os
import sys
import django
from django.utils import timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from attendance.models import RegisteredUser

today = timezone.localtime(timezone.now()).date()
new_users = RegisteredUser.objects.filter(created_at__date=today)

count = new_users.count()
print(f"COUNT FOR {today}: {count}")

if count > 0:
    for u in new_users:
        print(f"- {u.display_name} (Registered at: {timezone.localtime(u.created_at).strftime('%I:%M %p')})")
else:
    print("No users registered today.")
